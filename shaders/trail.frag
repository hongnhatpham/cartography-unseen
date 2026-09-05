#version 330

uniform sampler2D scene_depth;
uniform vec2 window_size;
uniform vec2 uv_scale;
uniform vec2 uv_offset;

in float opacity;
out vec4 frag_color;

void main() {
    vec2 source_uv = (gl_FragCoord.xy / window_size) * uv_scale + uv_offset;
    if (gl_FragCoord.z > texture(scene_depth, source_uv).r + 0.000001) discard;
    frag_color = vec4(vec3(232.0, 239.0, 223.0) / 255.0, opacity);
}
