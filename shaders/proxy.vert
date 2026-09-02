#version 330

in vec3 in_position;
in vec3 in_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_normal;
out vec3 world_position;

void main() {
    vec4 world = model * vec4(in_position, 1.0);
    world_position = world.xyz;
    world_normal = normalize(mat3(transpose(inverse(model))) * in_normal);
    gl_Position = projection * view * world;
}
