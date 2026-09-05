#version 330

in vec3 world_normal;
in vec3 world_position;
flat in vec3 material_color;

uniform vec3 camera_position;
uniform vec3 fog_color;
uniform vec3 zenith_color;
uniform vec3 nadir_color;
uniform float fog_distance;

out vec4 frag_color;

// Atmospheric depth. Full fade well inside the far plane, so the ends of the
// moving chunk window remain hidden while more geometry streams in.
const float FOG_START = 20.0;

// Background as a function of look direction. Kept identical to the copy in
// sky.frag so a form fading out lands exactly on the sky behind it.
vec3 background_color(vec3 dir) {
    float height = clamp(normalize(dir).y, -1.0, 1.0);
    if (height >= 0.0) {
        return mix(fog_color, zenith_color, pow(height, 0.80));
    }
    return mix(fog_color, nadir_color, pow(-height, 1.40));
}

// Hard four-step key light from a low raking angle, so upright faces split into
// lit and unlit halves and away-facing sides fall to roughly a quarter of their
// colour. The fill stays small on purpose: the conditioning needs real darks
// next to the pale panels, but a single unlit face can fill the whole frame, so
// it must keep a readable silhouette.
void main() {
    vec3 normal = normalize(world_normal);
    vec3 light_dir = normalize(vec3(0.58, 0.60, 0.34));
    float diffuse = max(dot(normal, light_dir), 0.0);
    float bands = floor(diffuse * 4.0 + 0.5) / 4.0;
    float key = 0.24 + 0.72 * bands;
    float sky = normal.y * 0.5 + 0.5;
    vec3 albedo = material_color;
    vec3 lit = albedo * (key + 0.08 + 0.16 * sky);

    vec3 view_ray = world_position - camera_position;
    float fog = clamp((length(view_ray) - FOG_START) / (fog_distance - FOG_START), 0.0, 1.0);
    fog = pow(fog, 1.25);
    frag_color = vec4(mix(lit, background_color(view_ray), fog), 1.0);
}
