#version 330

in vec3 world_normal;
in vec3 world_position;
flat in vec3 material_color;

uniform vec3 camera_position;
uniform vec3 fog_color;

out vec4 frag_color;

void main() {
    vec3 light_dir = normalize(vec3(0.45, 0.85, 0.25));
    float diffuse = max(dot(normalize(world_normal), light_dir), 0.0);
    float bands = floor((0.28 + diffuse * 0.72) * 6.0) / 6.0;
    float view_distance = distance(world_position.xz, camera_position.xz);
    float fog = clamp((view_distance - 95.0) / 80.0, 0.0, 1.0);
    vec3 lit = material_color * (0.38 + bands * 0.72);
    frag_color = vec4(mix(lit, fog_color, fog), 1.0);
}
