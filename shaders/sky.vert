#version 330

in vec2 in_position;

uniform vec3 camera_right;
uniform vec3 camera_up;
uniform vec3 camera_forward;
uniform float tan_half_fov;
uniform float aspect;

out vec3 view_dir;

// Full-screen quad drawn at the far plane. Each vertex carries the world-space
// ray through it, so the background can be a function of direction rather than
// of screen height: with no ground plane and no horizon the gradient has to
// hold in every direction the flier can look, including straight up and down.
void main() {
    gl_Position = vec4(in_position, 0.999, 1.0);
    view_dir = camera_forward
        + camera_right * (in_position.x * tan_half_fov * aspect)
        + camera_up * (in_position.y * tan_half_fov);
}
