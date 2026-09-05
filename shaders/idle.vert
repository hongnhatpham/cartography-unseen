#version 330
uniform vec2 window_size;
uniform vec4 rect;
in vec2 in_position;
out vec2 uv;
void main() {
    vec2 pixel = rect.xy + in_position * rect.zw;
    gl_Position = vec4(pixel / window_size * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
    uv = in_position;
}
