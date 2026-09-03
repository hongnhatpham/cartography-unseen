#version 330

in vec3 in_position;
in vec3 in_normal;
in vec4 instance_model_0;
in vec4 instance_model_1;
in vec4 instance_model_2;
in vec4 instance_model_3;
in vec3 instance_color;

uniform mat4 view;
uniform mat4 projection;

out vec3 world_normal;
out vec3 world_position;
flat out vec3 material_color;

void main() {
    mat4 model = mat4(
        instance_model_0,
        instance_model_1,
        instance_model_2,
        instance_model_3
    );
    vec4 world = model * vec4(in_position, 1.0);
    world_position = world.xyz;
    world_normal = normalize(mat3(model) * in_normal);
    material_color = instance_color;
    gl_Position = projection * view * world;
}
