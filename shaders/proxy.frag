#version 330

in vec3 world_normal;
in vec3 world_position;

uniform vec3 material_color;

out vec4 frag_color;

void main() {
    vec3 light_dir = normalize(vec3(0.45, 0.85, 0.25));
    float diffuse = max(dot(normalize(world_normal), light_dir), 0.0);
    float bands = floor((0.28 + diffuse * 0.72) * 6.0) / 6.0;
    float fog = clamp((length(world_position.xz) - 9.0) / 35.0, 0.0, 0.65);
    vec3 lit = material_color * (0.38 + bands * 0.72);
    vec3 fog_color = vec3(0.36, 0.50, 0.62);
    frag_color = vec4(mix(lit, fog_color, fog), 1.0);
}
