#version 330

in vec3 world_normal;
in vec3 world_position;
flat in vec3 material_color;

uniform vec3 camera_position;
uniform vec3 fog_color;

out vec4 frag_color;

// Two panel grids in world units and their swing either side of the face's own
// tone. The coarse grid structures a wall seen across the ravine; the fine one
// is what a face pressed against the eye still shows, and that near case is the
// one that fails. Both swing symmetrically so the world's mean brightness is
// unchanged: darkening the proxy overall only trades one failure for another.
const float PANEL_COARSE = 5.0;
const float PANEL_COARSE_SWING = 0.22;
const float PANEL_FINE = 1.25;
const float PANEL_FINE_SWING = 0.10;

float checker_sign(vec2 tangent, float size) {
    vec2 cell = floor(tangent / size);
    return mod(cell.x + cell.y, 2.0) * 2.0 - 1.0;
}

// Alternating light and dark panels on vertical faces, on world-space grids so
// the break is continuous across neighbouring instances. One large flat face
// filling the frame is an information vacuum, and the sampler completes it as a
// desk with a game controller on it; the same per-cell tone break already
// removed that failure on the ground plane. The grids are read on the two axes
// tangent to the face, so a wall carries columns and a slab carries panels
// instead of one flat value.
float panel_break(vec3 position, vec3 normal) {
    if (abs(normal.y) > 0.5) {
        return 1.0;
    }
    vec2 tangent = abs(normal.x) > abs(normal.z)
        ? vec2(position.z, position.y)
        : vec2(position.x, position.y);
    return 1.0
        + PANEL_COARSE_SWING * checker_sign(tangent, PANEL_COARSE)
        + PANEL_FINE_SWING * checker_sign(tangent, PANEL_FINE);
}

// Hard four-step key light from a low raking angle, so vertical cliff faces
// split into lit and unlit halves and away-facing sides fall to roughly a
// quarter of their colour. The fill stays small on purpose: the conditioning
// needs real darks next to the pale slabs, but at eye level an unlit face can
// fill the whole frame, so it must keep a readable silhouette.
void main() {
    vec3 normal = normalize(world_normal);
    vec3 light_dir = normalize(vec3(0.58, 0.60, 0.34));
    float diffuse = max(dot(normal, light_dir), 0.0);
    float bands = floor(diffuse * 4.0 + 0.5) / 4.0;
    float key = 0.16 + 0.80 * bands;
    float sky = normal.y * 0.5 + 0.5;
    vec3 albedo = material_color * panel_break(world_position, normal);
    vec3 lit = albedo * (key + 0.06 + 0.12 * sky);

    // Atmospheric depth starts a few steps ahead of the walker: at eye level
    // the only distance cue the conditioning can carry is forms fading toward
    // the sky, and without it the generated frame reads as flat.
    float view_distance = distance(world_position, camera_position);
    float fog = clamp((view_distance - 24.0) / 200.0, 0.0, 1.0);
    fog = pow(fog, 1.35) * 0.92;
    frag_color = vec4(mix(lit, fog_color, fog), 1.0);
}
