#include "glfw_vulkan.h"
int main(int ac, char **av)
{
    glfw::vulkan_app app;
    glfw::vulkan_window window(app, 640, 480);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        window.render([]() { ImGui::ShowDemoWindow(); });
    }
}
