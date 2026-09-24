/**
 * The parts of the SDK that need a filesystem.
 *
 * Kept out of the main entry so a browser bundle never has to resolve `node:fs`.
 */

export {
  AGENT_FILE,
  AGENT_STAMP,
  GUARDRAIL_FILE,
  INSTRUCTIONS_FILE,
  KNOWLEDGE_DIR,
  KNOWLEDGE_URLS_FILE,
  SKILLS_DIR,
  loadFolder,
  parseDeclaration,
  parsePages,
  parseSkill,
} from "./folder.js";
