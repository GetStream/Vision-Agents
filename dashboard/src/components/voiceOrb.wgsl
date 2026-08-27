import { fbmSimplex3d } from "@vgpu/wgsl-std/noise/simplex";

struct Params {
  // tint is the colour of whoever has the floor.
  tint: vec3f,
  time: f32,
  // energy is how hard the orb is working: it drives the fast detail and the rings.
  energy: f32,
  // presence is how awake the call is, which survives the gaps between turns.
  presence: f32,
  texel: vec2f,
}

@group(0) @binding(0) var<uniform> params: Params;

const BACKDROP = vec3f(0.039, 0.039, 0.051);

// The radius the orb returns to, in half-heights of the panel.
const REST = 0.5;

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let resolution = vec2f(1.0) / max(params.texel, vec2f(0.000001));
  let aspect = max(resolution.x, 1.0) / max(resolution.y, 1.0);

  var p = uv - vec2f(0.5);
  p.x *= aspect;
  p *= 2.0;

  let dist = length(p);
  let time = params.time;
  let tint = params.tint;

  // The rim is displaced by noise sampled around a circle rather than across the
  // plane, so the outline stays one smooth closed curve however hard it moves.
  let around = vec2f(p.x, p.y) / max(dist, 0.0001);
  let breath = fbmSimplex3d(vec3f(around * 1.2, time * 0.22), 3, 2.0, 0.5);
  let chatter = fbmSimplex3d(vec3f(around * 1.8, time * 1.3), 2, 2.0, 0.5);

  let radius = REST
    + breath * 0.07 * (0.4 + params.presence)
    + chatter * 0.05 * params.energy;

  // Everything below is a function of the distance past the rim, so the body, the
  // rim light and the glow stay consistent as the rim moves.
  let edge = dist - radius;
  let body = smoothstep(0.02, -0.20, edge);
  let rim = exp(-abs(edge) * 14.0);

  // The light thrown past the rim is measured from the resting radius instead of
  // the moving one: taking it from the wobble turns every bulge into a sunbeam.
  let outside = max(dist - REST, 0.0);
  let glow = exp(-outside * 3.0) * (0.18 + params.energy * 0.65);

  // Rings leaving the orb while it talks.
  let wave = sin((dist - time * 0.45) * 20.0) * 0.5 + 0.5;
  let rings = wave * exp(-outside * 3.5) * params.energy * 0.22;

  // A slow band of light across the panel, so the empty width still belongs to
  // the same picture.
  let drift = fbmSimplex3d(vec3f(p.x * 0.25, p.y * 1.4, time * 0.1), 3, 2.0, 0.5);
  let veil = exp(-abs(p.y - drift * 0.5) * 2.2) * (0.03 + params.presence * 0.09);

  var colour = BACKDROP;
  colour += tint * veil;
  colour += tint * body * (0.16 + params.presence * 0.22);
  colour += tint * rim * (0.7 + params.energy * 1.0);
  colour += tint * glow * 0.4;
  colour += tint * rings;

  // A brighter core, offset upward, keeps the body from reading as a flat disc.
  let core = exp(-length(p - vec2f(0.0, 0.16)) * 5.0);
  colour += tint * core * (0.10 + params.energy * 0.30);

  // The vignette is measured in panel corners rather than orb radii, so it hugs the
  // edges of a wide strip instead of swallowing everything either side of the orb.
  let corner = length(uv - vec2f(0.5)) * 2.0;
  colour *= 1.0 - smoothstep(0.75, 1.45, corner) * 0.55;

  return vec4f(max(colour, BACKDROP * 0.65), 1.0);
}
