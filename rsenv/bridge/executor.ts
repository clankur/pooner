// Bridge between Python GRPO training loop and rs-sdk.
// Reads JSON commands from stdin, executes via BotActions, writes JSON results to stdout.
// All stderr output is for logging only; stdout is the protocol channel.

// Redirect console.log/warn/info to stderr so rs-sdk internal logging
// doesn't corrupt the JSON protocol on stdout.
const _origLog = console.log;
const _origWarn = console.warn;
const _origInfo = console.info;
console.log = (...args: unknown[]) => process.stderr.write(args.map(String).join(" ") + "\n");
console.warn = (...args: unknown[]) => process.stderr.write("[warn] " + args.map(String).join(" ") + "\n");
console.info = (...args: unknown[]) => process.stderr.write("[info] " + args.map(String).join(" ") + "\n");

import { BotSDK, deriveGatewayUrl } from "./rs-sdk/sdk/index";
import { BotActions } from "./rs-sdk/sdk/actions";
import type {
  BotWorldState,
  SkillState,
  InventoryItem,
  NearbyNpc,
  NearbyLoc,
  GroundItem,
} from "./rs-sdk/sdk/types";

// ─── Types ───────────────────────────────────────────────────────────────

interface Command {
  type: "getState" | "action" | "reset";
  name?: string;
  arguments?: Record<string, unknown>;
}

interface StateResponse {
  type: "state";
  data: {
    tick: number;
    inGame: boolean;
    position: {
      x: number;
      z: number;
      worldX: number;
      worldZ: number;
      level: number;
    };
    skills: { name: string; level: number; baseLevel: number; experience: number }[];
    inventory: { slot: number; id: number; name: string; count: number }[];
    equipment: { slot: number; id: number; name: string; count: number }[];
    nearbyNpcs: {
      index: number;
      name: string;
      combatLevel: number;
      x: number;
      z: number;
      distance: number;
      hp: number;
      maxHp: number;
      inCombat: boolean;
      options: string[];
    }[];
    nearbyLocs: {
      id: number;
      name: string;
      x: number;
      z: number;
      distance: number;
      options: string[];
    }[];
    groundItems: {
      id: number;
      name: string;
      count: number;
      x: number;
      z: number;
      distance: number;
    }[];
    hp: number;
    maxHp: number;
    inCombat: boolean;
  };
}

interface ActionResultResponse {
  type: "result";
  success: boolean;
  observation: string;
  xpGained?: Record<string, number>;
  inventoryChanges?: { name: string; delta: number }[];
  reason?: string;
}

interface ErrorResponse {
  type: "error";
  message: string;
}

type Response = StateResponse | ActionResultResponse | ErrorResponse;

// ─── Helpers ─────────────────────────────────────────────────────────────

function log(msg: string): void {
  process.stderr.write(`[bridge] ${msg}\n`);
}

function send(response: Response): void {
  process.stdout.write(JSON.stringify(response) + "\n");
}

function formatState(state: BotWorldState): StateResponse {
  const player = state.player;
  return {
    type: "state",
    data: {
      tick: state.tick,
      inGame: state.inGame,
      position: {
        x: player?.x ?? 0,
        z: player?.z ?? 0,
        worldX: player?.worldX ?? 0,
        worldZ: player?.worldZ ?? 0,
        level: player?.level ?? 0,
      },
      skills: state.skills.map((s) => ({
        name: s.name,
        level: s.level,
        baseLevel: s.baseLevel,
        experience: s.experience,
      })),
      inventory: state.inventory.map((i) => ({
        slot: i.slot,
        id: i.id,
        name: i.name,
        count: i.count,
      })),
      equipment: state.equipment.map((i) => ({
        slot: i.slot,
        id: i.id,
        name: i.name,
        count: i.count,
      })),
      nearbyNpcs: state.nearbyNpcs.map((n) => ({
        index: n.index,
        name: n.name,
        combatLevel: n.combatLevel,
        x: n.x,
        z: n.z,
        distance: n.distance,
        hp: n.hp,
        maxHp: n.maxHp,
        inCombat: n.inCombat,
        options: n.options,
      })),
      nearbyLocs: state.nearbyLocs.map((l) => ({
        id: l.id,
        name: l.name,
        x: l.x,
        z: l.z,
        distance: l.distance,
        options: l.options,
      })),
      groundItems: state.groundItems.map((g) => ({
        id: g.id,
        name: g.name,
        count: g.count,
        x: g.x,
        z: g.z,
        distance: g.distance,
      })),
      hp: player?.hp ?? 0,
      maxHp: player?.maxHp ?? 0,
      inCombat: player?.combat?.inCombat ?? false,
    },
  };
}

/** Diff two inventory snapshots to find what changed. */
function diffInventory(
  before: InventoryItem[],
  after: InventoryItem[]
): { name: string; delta: number }[] {
  const countsBefore = new Map<string, number>();
  const countsAfter = new Map<string, number>();

  for (const item of before) {
    countsBefore.set(item.name, (countsBefore.get(item.name) ?? 0) + item.count);
  }
  for (const item of after) {
    countsAfter.set(item.name, (countsAfter.get(item.name) ?? 0) + item.count);
  }

  const allNames = new Set([...countsBefore.keys(), ...countsAfter.keys()]);
  const changes: { name: string; delta: number }[] = [];

  for (const name of allNames) {
    const delta = (countsAfter.get(name) ?? 0) - (countsBefore.get(name) ?? 0);
    if (delta !== 0) {
      changes.push({ name, delta });
    }
  }

  return changes;
}

/** Diff two skill snapshots to find XP changes. */
function diffSkills(
  before: SkillState[],
  after: SkillState[]
): Record<string, number> {
  const xpGained: Record<string, number> = {};
  for (const skillAfter of after) {
    const skillBefore = before.find((s) => s.name === skillAfter.name);
    const delta = skillAfter.experience - (skillBefore?.experience ?? 0);
    if (delta > 0) {
      xpGained[skillAfter.name] = delta;
    }
  }
  return xpGained;
}

/** Build a human-readable observation from action results and state diffs. */
function buildObservation(
  actionName: string,
  message: string,
  xpGained: Record<string, number>,
  invChanges: { name: string; delta: number }[]
): string {
  const parts: string[] = [];

  for (const [skill, xp] of Object.entries(xpGained)) {
    parts.push(`+${xp} ${skill} XP`);
  }

  for (const change of invChanges) {
    if (change.delta > 0) {
      parts.push(`Got ${change.name}${change.delta > 1 ? ` x${change.delta}` : ""}.`);
    } else {
      parts.push(
        `Lost ${change.name}${change.delta < -1 ? ` x${Math.abs(change.delta)}` : ""}.`
      );
    }
  }

  // Fall back to the SDK message if we have no specific observations
  if (parts.length === 0) {
    parts.push(message);
  }

  return parts.join(" ");
}

// ─── Action dispatch ─────────────────────────────────────────────────────

async function executeAction(
  bot: BotActions,
  sdk: BotSDK,
  name: string,
  args: Record<string, unknown>
): Promise<ActionResultResponse> {
  const stateBefore = sdk.getState();
  const skillsBefore = stateBefore?.skills ?? [];
  const invBefore = stateBefore?.inventory ?? [];

  let success = false;
  let message = "";
  let reason: string | undefined;

  try {
    switch (name) {
      case "chopTree": {
        const target = (args.tree_type as string) ?? "Tree";
        const result = await bot.chopTree(new RegExp(target, "i"));
        success = result.success;
        message = result.message;
        break;
      }

      case "walkTo": {
        const x = args.x as number;
        const z = (args.z ?? args.y) as number;
        const result = await bot.walkTo(x, z);
        success = result.success;
        message = result.message;
        break;
      }

      case "attackNpc": {
        const npcName = args.npc_name as string;
        const result = await bot.attackNpc(new RegExp(npcName, "i"));
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "dropInventory": {
        const itemName = args.item as string;
        const item = sdk.findInventoryItem(new RegExp(itemName, "i"));
        if (!item) {
          return {
            type: "result",
            success: false,
            observation: `No ${itemName} in inventory.`,
            reason: "item_not_found",
          };
        }
        // Drop via the item's "Drop" option
        const dropOpt = item.optionsWithIndex.find((o) => /drop/i.test(o.text));
        if (!dropOpt) {
          return {
            type: "result",
            success: false,
            observation: `Cannot drop ${item.name}.`,
            reason: "no_drop_option",
          };
        }
        const result = await sdk.sendUseItem(item.slot, dropOpt.opIndex);
        success = result.success;
        message = result.success ? `Dropped ${item.name}.` : result.message;
        // Wait a tick for state to settle
        await sdk.waitForTicks(1);
        break;
      }

      case "openBank": {
        const result = await bot.openBank();
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "bankDeposit": {
        const itemName = args.item as string;
        const quantity = (args.quantity as number) ?? -1;
        const result = await bot.depositItem(
          new RegExp(itemName, "i"),
          quantity
        );
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "bankWithdraw": {
        const itemName = args.item as string;
        const quantity = (args.quantity as number) ?? 1;
        const result = await bot.withdrawItem(
          new RegExp(itemName, "i"),
          quantity
        );
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "mineRock": {
        const rockType = (args.rock_type as string) ?? "Rock";
        // Mining is done via interactLoc with the "Mine" option
        const result = await bot.interactLoc(new RegExp(rockType, "i"), /mine/i);
        success = result.success;
        message = result.message;
        reason = result.reason;
        if (success) {
          // Wait for the mining animation to complete and item to appear
          try {
            await sdk.waitForCondition(
              (s) => s.player?.animId === -1,
              15000
            );
          } catch {
            // Timeout is fine, we still got the interaction started
          }
        }
        break;
      }

      case "catchFish": {
        const spotType = (args.spot_type as string) ?? "Fishing spot";
        // Fishing spots are NPCs in RS, interact with them
        const result = await bot.interactNpc(new RegExp(spotType, "i"), /net|lure|bait|cage|harpoon/i);
        success = result.success;
        message = result.message;
        reason = result.reason;
        if (success) {
          try {
            await sdk.waitForCondition(
              (s) => s.player?.animId === -1,
              15000
            );
          } catch {}
        }
        break;
      }

      case "cookItem": {
        const itemName = args.item as string;
        const result = await bot.useItemOnLoc(
          new RegExp(itemName, "i"),
          /range|fire|stove/i
        );
        success = result.success;
        message = result.message;
        reason = result.reason;
        if (success) {
          try {
            await sdk.waitForCondition(
              (s) => s.player?.animId === -1,
              10000
            );
          } catch {}
        }
        break;
      }

      case "buryBones": {
        const bones = sdk.findInventoryItem(/bones/i);
        if (!bones) {
          return {
            type: "result",
            success: false,
            observation: "No bones in inventory.",
            reason: "item_not_found",
          };
        }
        const buryOpt = bones.optionsWithIndex.find((o) => /bury/i.test(o.text));
        if (!buryOpt) {
          return {
            type: "result",
            success: false,
            observation: "Cannot bury this item.",
            reason: "no_bury_option",
          };
        }
        const result = await sdk.sendUseItem(bones.slot, buryOpt.opIndex);
        success = result.success;
        message = result.success ? "Buried bones." : result.message;
        await sdk.waitForTicks(2);
        break;
      }

      case "equipItem": {
        const itemName = args.item as string;
        const result = await bot.equipItem(new RegExp(itemName, "i"));
        success = result.success;
        message = result.message;
        break;
      }

      case "craftItem": {
        const itemName = args.item as string;
        // Try leather crafting first, fall back to generic craft
        const result = await bot.craftLeather(itemName);
        success = result.success;
        message = result.message;
        break;
      }

      case "pickpocket": {
        const npcName = args.npc_name as string;
        const result = await bot.pickpocketNpc(new RegExp(npcName, "i"));
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "castSpell": {
        const target = args.target as string;
        // Use Wind Strike (component 1152) as default combat spell
        const spellComponent = 1152;
        const npc = sdk.findNearbyNpc(new RegExp(target, "i"));
        if (!npc) {
          return {
            type: "result",
            success: false,
            observation: `No ${target} found nearby.`,
            reason: "npc_not_found",
          };
        }
        const result = await bot.castSpellOnNpc(npc, spellComponent);
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "eatFood": {
        const itemName = args.item as string;
        const result = await bot.eatFood(new RegExp(itemName, "i"));
        success = result.success;
        message = result.message;
        break;
      }

      case "pickupItem": {
        const itemName = args.item as string;
        const result = await bot.pickupItem(new RegExp(itemName, "i"));
        success = result.success;
        message = result.message;
        reason = result.reason;
        break;
      }

      case "fletchLogs": {
        const product = args.product as string | undefined;
        const result = await bot.fletchLogs(product);
        success = result.success;
        message = result.message;
        break;
      }

      case "talkTo": {
        const npcName = args.npc_name as string;
        const result = await bot.talkTo(new RegExp(npcName, "i"));
        success = result.success;
        message = result.message;
        break;
      }

      case "smithItem": {
        const product = (args.product as string) ?? "dagger";
        const result = await bot.smithAtAnvil(product);
        success = result.success;
        message = result.message;
        break;
      }

      default:
        return {
          type: "error",
          message: `Unknown action: ${name}`,
        } as ErrorResponse;
    }
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return {
      type: "result",
      success: false,
      observation: `Action failed: ${errMsg}`,
      reason: "exception",
    };
  }

  // Compute diffs
  const stateAfter = sdk.getState();
  const skillsAfter = stateAfter?.skills ?? [];
  const invAfter = stateAfter?.inventory ?? [];

  const xpGained = diffSkills(skillsBefore, skillsAfter);
  const invChanges = diffInventory(invBefore, invAfter);
  const observation = buildObservation(name, message, xpGained, invChanges);

  return {
    type: "result",
    success,
    observation,
    xpGained: Object.keys(xpGained).length > 0 ? xpGained : undefined,
    inventoryChanges: invChanges.length > 0 ? invChanges : undefined,
    reason,
  };
}

// ─── Main loop ───────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // Load credentials from bot.env if available
  const username = process.env.RS_BOT_USERNAME ?? "grpobot1";
  const botEnvPath = `${import.meta.dir}/rs-sdk/bots/${username}/bot.env`;
  let password = process.env.RS_BOT_PASSWORD ?? "";
  try {
    const envFile = await Bun.file(botEnvPath).text();
    for (const line of envFile.split("\n")) {
      const [key, ...vals] = line.split("=");
      if (key.trim() === "PASSWORD") password = vals.join("=").trim();
    }
    log(`Loaded credentials from ${botEnvPath}`);
  } catch {
    log(`No bot.env found at ${botEnvPath}, using env vars`);
  }

  const gatewayUrl = deriveGatewayUrl(process.env.RS_GATEWAY ?? "");

  // Detect headless: no DISPLAY on Linux, or HEADLESS=true env var
  const isHeadless = process.env.HEADLESS === "true" || process.env.HEADLESS === "1"
    || (process.platform === "linux" && !process.env.DISPLAY);

  log(`Connecting to gateway: ${gatewayUrl}`);
  log(`Bot username: ${username} (headless: ${isHeadless})`);

  const sdk = new BotSDK({
    botUsername: username,
    password,
    gatewayUrl,
    autoLaunchBrowser: !isHeadless,
    autoReconnect: true,
    showChat: false,
  });

  const bot = new BotActions(sdk);

  try {
    await sdk.connect();
    log("Connected to gateway");
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    send({ type: "error", message: `Failed to connect: ${errMsg}` });
    process.exit(1);
  }

  // On headless systems, launch bot client via puppeteer
  if (isHeadless) {
    log("Headless mode: launching bot client via puppeteer...");
    const { launchBotBrowser } = await import("./rs-sdk/sdk/test/utils/browser");
    await launchBotBrowser(username, { headless: true, useSharedBrowser: false });
    log("Puppeteer bot client launched");
  }

  // Wait for the bot client to be in-game
  log("Waiting for bot to enter game world...");
  const maxWaitMs = 60_000;
  const pollMs = 1_000;
  const startTime = Date.now();
  while (Date.now() - startTime < maxWaitMs) {
    const s = sdk.getState();
    if (s?.inGame && s.player) {
      log(`Bot is in-game at (${s.player.x}, ${s.player.z})`);
      break;
    }
    await new Promise((r) => setTimeout(r, pollMs));
  }

  // Skip tutorial if on Tutorial Island
  log("Skipping tutorial if needed...");
  await bot.skipTutorial();
  log("Tutorial check complete");

  // Signal readiness by sending initial state
  const initialState = sdk.getState();
  if (initialState?.inGame) {
    send(formatState(initialState));
  } else {
    send({ type: "error", message: "Bot not in game after 60s wait" });
  }

  // Read stdin line by line
  const decoder = new TextDecoder();
  const reader = Bun.stdin.stream().getReader();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      log("stdin closed, shutting down");
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // Keep the last partial line in the buffer
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let cmd: Command;
      try {
        cmd = JSON.parse(trimmed) as Command;
      } catch {
        send({ type: "error", message: `Invalid JSON: ${trimmed}` });
        continue;
      }

      if (cmd.type === "getState") {
        const state = sdk.getState();
        if (state) {
          send(formatState(state));
        } else {
          send({ type: "error", message: "No game state available" });
        }
      } else if (cmd.type === "action") {
        if (!cmd.name) {
          send({ type: "error", message: "Action missing 'name' field" });
          continue;
        }
        const result = await executeAction(
          bot,
          sdk,
          cmd.name,
          cmd.arguments ?? {}
        );
        send(result);
      } else if (cmd.type === "reset") {
        // Reset means disconnect and reconnect, getting a fresh bot state
        log("Reset requested");
        sdk.disconnect();
        try {
          await sdk.connect();
          const state = sdk.getState();
          if (state) {
            send(formatState(state));
          } else {
            send({ type: "error", message: "No state after reset" });
          }
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Reset failed: ${errMsg}` });
        }
      } else {
        send({
          type: "error",
          message: `Unknown command type: ${(cmd as any).type}`,
        });
      }
    }
  }

  // Cleanup
  sdk.disconnect();
  log("Disconnected");
}

main().catch((err) => {
  log(`Fatal error: ${err}`);
  process.exit(1);
});
