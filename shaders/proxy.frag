#version 330

in vec3 world_normal;
in vec3 world_position;
flat in vec3 material_color;

uniform vec3 camera_position;
uniform vec3 fog_color;

out vec4 frag_color;

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
    vec3 lit = material_color * (key + 0.06 + 0.12 * sky);

    // Atmospheric depth starts a few steps ahead of the walker: at eye level
    // the only distance cue the conditioning can carry is forms fading toward
    // the sky, and without it the generated frame reads as flat.
    float view_distance = distance(world_position, camera_position);
    float fog = clamp((view_distance - 24.0) / 200.0, 0.0, 1.0);
    fog = pow(fog, 1.35) * 0.92;
    frag_color = vec4(mix(lit, fog_color, fog), 1.0);
}
