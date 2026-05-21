// Minimal ambient declarations for the node builtins used by validate.ts.
// The parser_evolver package keeps @types/node out of devDependencies on
// purpose — the prototype is meant to be readable without a typings tree.
// This shim covers only what the validator imports, nothing more.

declare module "node:fs" {
  export function readFileSync(path: string, encoding: "utf8"): string;
  export function existsSync(path: string): boolean;
}

declare module "node:path" {
  export function dirname(path: string): string;
  export function resolve(...parts: string[]): string;
}

declare module "node:url" {
  export function fileURLToPath(url: string): string;
}

declare const process: { exit(code?: number): never };
