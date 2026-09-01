import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Dockerfile copies .next/standalone, which needs the traced server bundle.
  output: "standalone",
  turbopack: {
    rules: {
      // The orb's shader imports WGSL modules, and the loader resolves that graph
      // at build time. `as` is required so Turbopack treats the output as JS.
      "*.wgsl": {
        loaders: ["@vgpu/wgsl/loader-webpack"],
        as: "*.js",
      },
    },
  },
};

export default nextConfig;
