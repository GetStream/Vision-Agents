/**
 * The parts of the SDK that need a filesystem.
 *
 * Kept out of the main entry so a browser bundle never has to resolve `node:fs`.
 */

export {
  GUARDRAIL_FILE,
  INSTRUCTIONS_FILE,
  KNOWLEDGE_DIR,
  KNOWLEDGE_URLS_FILE,
  SKILLS_DIR,
  loadFolder,
  parsePages,
  parseSkill,
} from "./folder.js";
