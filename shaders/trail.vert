#version 330

uniform mat4 view_projection;
uniform vec2 uv_scale;
uniform vec2 uv_offset;

in vec3 in_position;
in float in_opacity;
out float vertex_opacity;

void main() {
    vec4 clip = view_projection * vec4(in_position - vec3(0.0, 0.7, 0.0), 1.0);
    // Invert the image's cover crop so the filament and proxy stay registered.
    clip.xy = (clip.xy + clip.w * (vec2(1.0) - 2.0 * uv_offset - uv_scale)) / uv_scale;
    gl_Position = clip;
    vertex_opacity = in_opacity;
}
