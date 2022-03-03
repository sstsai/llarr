#pragma once
#include <vulkan/vulkan.hpp>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_glfw.h>
#include <GLFW/glfw3.h>
#include <fmt/core.h>
#include <optional>
#include <vector>
#include <limits>
#include <source_location>
#include <string_view>
#include <exception>
namespace glfw {
struct release_instance {
    struct pointer {
        int x;
        pointer(int t) : x(t) {}
        pointer(std::nullptr_t = nullptr) : x(0) {}
        explicit operator bool() const { return !!x; }
        friend bool operator==(pointer lhs, pointer rhs)
        {
            return lhs.x == rhs.x;
        }
        friend bool operator!=(pointer lhs, pointer rhs)
        {
            return !(lhs == rhs);
        }
    };
    void operator()(pointer) const { glfwTerminate(); }
};
using instance = std::unique_ptr<void, release_instance>;
struct destroy_window {
    using pointer = GLFWwindow *;
    void operator()(pointer p) const { glfwDestroyWindow(p); }
};
using window = std::unique_ptr<void, destroy_window>;
auto make_window(int width, int height, char const *title = "",
                 GLFWmonitor *monitor = nullptr, GLFWwindow *share = nullptr)
    -> window
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    return window(glfwCreateWindow(width, height, title, monitor, share));
}
inline void terminate_on_error(
    vk::Result result,
    std::source_location location = std::source_location::current())
{
    if (result != vk::Result::eSuccess) {
        fmt::print("file: {}({}:{}) `{}`: {}\n", location.file_name(),
                   location.line(), location.column(), location.function_name(),
                   to_string(result));
        std::terminate();
    }
}
template <typename T>
inline auto value_or_terminate(
    vk::ResultValue<T> result_value,
    std::source_location location = std::source_location::current()) -> T
{
    terminate_on_error(result_value.result);
    return std::move(result_value.value);
}
inline auto vulkan_instance() -> vk::UniqueInstance
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init(
        (PFN_vkGetInstanceProcAddr)glfwGetInstanceProcAddress(
            {}, "vkGetInstanceProcAddr"));
    uint32_t count;
    auto extensions = glfwGetRequiredInstanceExtensions(&count);
    auto instance   = value_or_terminate(vk::createInstanceUnique(
          vk::InstanceCreateInfo{.enabledExtensionCount   = count,
                                 .ppEnabledExtensionNames = extensions}));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    return instance;
}
inline auto vulkan_queue_selection(vk::Instance instance)
    -> std::optional<ImGui_ImplVulkan_InitInfo>
{
    for (auto physical_device :
         value_or_terminate(instance.enumeratePhysicalDevices())) {
        auto queues = physical_device.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queues.size(); i++) {
            if (queues[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                if (glfwGetPhysicalDevicePresentationSupport(
                        instance, physical_device, i)) {
                    return ImGui_ImplVulkan_InitInfo{.Instance = instance,
                                                     .PhysicalDevice =
                                                         physical_device,
                                                     .QueueFamily = i};
                }
            }
        }
    }
    return {};
}
inline auto vulkan_surface(vk::Instance instance, GLFWwindow *window)
    -> vk::UniqueSurfaceKHR
{
    for (auto physical_device :
         value_or_terminate(instance.enumeratePhysicalDevices())) {
        auto queues = physical_device.getQueueFamilyProperties();
        for (auto i = 0; i < queues.size(); i++) {
            if (glfwGetPhysicalDevicePresentationSupport(instance,
                                                         physical_device, i)) {
                VkSurfaceKHR surface;
                auto result = glfwCreateWindowSurface(instance, window, nullptr,
                                                      &surface);
                return vk::UniqueSurfaceKHR(surface, instance);
            }
        }
    }
    return {};
}
inline auto vulkan_queue(vk::PhysicalDevice physical_device,
                         uint32_t queue_family_index) -> vk::UniqueDevice
{
    auto queue_priority    = 1.0f;
    auto device_extensions = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    auto queue_create_info =
        vk::DeviceQueueCreateInfo{.queueFamilyIndex = queue_family_index,
                                  .queueCount       = 1,
                                  .pQueuePriorities = &queue_priority};
    auto device = value_or_terminate(physical_device.createDeviceUnique(
        vk::DeviceCreateInfo{.queueCreateInfoCount    = 1,
                             .pQueueCreateInfos       = &queue_create_info,
                             .enabledExtensionCount   = 1,
                             .ppEnabledExtensionNames = &device_extensions}));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    return device;
}
inline auto vulkan_pool(vk::Device device) -> vk::UniqueDescriptorPool
{
    constexpr auto pool_sizes = std::array<vk::DescriptorPoolSize, 11>{{
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000},
    }};
    return device.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{
        .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets       = 1000 * pool_sizes.size(),
        .poolSizeCount = pool_sizes.size(),
        .pPoolSizes    = pool_sizes.data()});
}
inline auto vulkan_render_pass(vk::Device device, vk::Format format)
    -> vk::UniqueRenderPass
{
    auto attachment_description = vk::AttachmentDescription{
        .format         = format,
        .loadOp         = vk::AttachmentLoadOp::eClear,
        .storeOp        = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout  = vk::ImageLayout::eUndefined,
        .finalLayout    = vk::ImageLayout::ePresentSrcKHR};
    auto attachment_reference = vk::AttachmentReference{
        .layout = vk::ImageLayout::eColorAttachmentOptimal};
    auto subpass_description = vk::SubpassDescription{
        .colorAttachmentCount = 1, .pColorAttachments = &attachment_reference};
    auto subpass_dependency = vk::SubpassDependency{
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite};
    return device.createRenderPassUnique(
        vk::RenderPassCreateInfo{.attachmentCount = 1,
                                 .pAttachments    = &attachment_description,
                                 .subpassCount    = 1,
                                 .pSubpasses      = &subpass_description,
                                 .dependencyCount = 1,
                                 .pDependencies   = &subpass_dependency});
}
inline auto vulkan_image_view(vk::Device device, vk::Image image,
                              vk::Format format) -> vk::UniqueImageView
{
    return device.createImageViewUnique(vk::ImageViewCreateInfo{
        .image            = image,
        .viewType         = vk::ImageViewType::e2D,
        .format           = format,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1}});
}
inline auto vulkan_frame_buffer(vk::Device device, vk::RenderPass render_pass,
                                vk::ImageView image_view,
                                vk::Extent2D image_extent)
    -> vk::UniqueFramebuffer
{
    return device.createFramebufferUnique(
        vk::FramebufferCreateInfo{.renderPass      = render_pass,
                                  .attachmentCount = 1,
                                  .pAttachments    = &image_view,
                                  .width           = image_extent.width,
                                  .height          = image_extent.height,
                                  .layers          = 1});
}
inline auto vulkan_command_pool(vk::Device device, uint32_t queue_family_index)
    -> vk::UniqueCommandPool
{
    return device.createCommandPoolUnique(vk::CommandPoolCreateInfo{
        .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queue_family_index});
}
inline auto vulkan_command_buffer(vk::Device device,
                                  vk::CommandPool command_pool)
    -> vk::UniqueCommandBuffer
{
    return std::move(value_or_terminate(
        device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
            .commandPool = command_pool, .commandBufferCount = 1}))[0]);
}
struct vulkan_sync {
public:
    vk::Device device;
    vk::CommandPool command_pool;
    vk::Queue queue;
    vk::UniqueSemaphore image_acquired_semaphore;
    vk::UniqueSemaphore render_complete_semaphore;
    vk::UniqueFence fence;
    vk::UniqueCommandBuffer command_buffer;

public:
    vulkan_sync(vk::Device device, vk::CommandPool command_pool,
                vk::Queue queue)
        : device(device), command_pool(command_pool), queue(queue),
          image_acquired_semaphore(device.createSemaphoreUnique({})),
          render_complete_semaphore(device.createSemaphoreUnique({})),
          fence(device.createFenceUnique(vk::FenceCreateInfo{
              .flags = vk::FenceCreateFlagBits::eSignaled})),
          command_buffer(std::move(value_or_terminate(
              device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
                  .commandPool = command_pool, .commandBufferCount = 1}))[0]))
    {}
    operator vk::CommandBuffer() const { return *command_buffer; }
    auto acquire(vk::SwapchainKHR swapchain)
    {
        return device.acquireNextImageKHR(
            swapchain, std::numeric_limits<uint64_t>::max(),
            *image_acquired_semaphore, vk::Fence());
    }
    template <typename Fn> void command(vk::Fence fence, Fn &&fn)
    {
        device.resetCommandPool(command_pool);
        terminate_on_error(command_buffer->begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
        std::forward<Fn>(fn)(*command_buffer);
        terminate_on_error(command_buffer->end());
        terminate_on_error(
            queue.submit(vk::SubmitInfo{.commandBufferCount = 1,
                                        .pCommandBuffers    = &*command_buffer},
                         fence));
    }
    template <typename Fn>
    void render(vk::RenderPassBeginInfo const &begin_info, Fn &&fn)
    {
        terminate_on_error(device.waitForFences(
            *fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
        terminate_on_error(device.resetFences(*fence));
        device.resetCommandPool(command_pool);
        terminate_on_error(command_buffer->begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
        command_buffer->beginRenderPass(begin_info,
                                        vk::SubpassContents::eInline);
        std::forward<Fn>(fn)(*command_buffer);
        command_buffer->endRenderPass();
        terminate_on_error(command_buffer->end());
        vk::PipelineStageFlags pipeline_stage_flag =
            vk::PipelineStageFlagBits::eColorAttachmentOutput;
        terminate_on_error(queue.submit(
            vk::SubmitInfo{.waitSemaphoreCount   = 1,
                           .pWaitSemaphores      = &*image_acquired_semaphore,
                           .pWaitDstStageMask    = &pipeline_stage_flag,
                           .commandBufferCount   = 1,
                           .pCommandBuffers      = &*command_buffer,
                           .signalSemaphoreCount = 1,
                           .pSignalSemaphores    = &*render_complete_semaphore},
            *fence));
    }
    auto present(vk::SwapchainKHR swapchain, uint32_t image_index)
    {
        return queue.presentKHR(
            vk::PresentInfoKHR{.waitSemaphoreCount = 1,
                               .pWaitSemaphores = &*render_complete_semaphore,
                               .swapchainCount  = 1,
                               .pSwapchains     = &swapchain,
                               .pImageIndices   = &image_index});
    }
};
struct vulkan_app {
private:
    instance context          = glfw::instance(glfwInit());
    vk::UniqueInstance vulkan = vulkan_instance();

public:
    vulkan_app() { ImGui_ImplVulkan_LoadFunctions(loader_func, this); }
    operator vk::Instance() const { return *vulkan; }

private:
    static auto loader_func(char const *function_name, void *user_data)
        -> PFN_vkVoidFunction
    {
        auto *app = static_cast<vulkan_app *>(user_data);
        return glfwGetInstanceProcAddress(*(app->vulkan), function_name);
    }
};
struct vulkan_window {
public:
    vk::SwapchainCreateInfoKHR swapchain_info{
        .minImageCount    = 2,
        .imageFormat      = vk::Format::eB8G8R8A8Unorm,
        .imageColorSpace  = vk::ColorSpaceKHR::eSrgbNonlinear,
        .imageArrayLayers = 1,
        .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
        .presentMode      = vk::PresentModeKHR::eFifo,
        .clipped          = VK_TRUE};
    window win;
    ImGui_ImplVulkan_InitInfo imgui_vulkan_init;
    vk::UniqueDevice device;
    vk::UniqueDescriptorPool pool;
    vk::UniqueRenderPass render_pass;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueSwapchainKHR swapchain;
    vk::UniqueCommandPool command_pool;
    struct frame {
        vulkan_sync sync;
        vk::UniqueImageView image_view;
        vk::UniqueFramebuffer frame_buffer;
    };
    std::vector<frame> frames;
    unsigned index{};
    bool rebuild_swapchain{};

public:
    vulkan_window(vk::Instance instance, int width, int height,
                  char const *title = "", GLFWmonitor *monitor = nullptr,
                  GLFWwindow *share = nullptr)
        : win(make_window(width, height, title, monitor, share)),
          imgui_vulkan_init(*vulkan_queue_selection(instance)),
          device(vulkan_queue(imgui_vulkan_init.PhysicalDevice,
                              imgui_vulkan_init.QueueFamily)),
          pool(vulkan_pool(*device)),
          render_pass(vulkan_render_pass(*device, swapchain_info.imageFormat)),
          surface(vulkan_surface(instance, *this)),
          command_pool(
              vulkan_command_pool(*device, imgui_vulkan_init.QueueFamily))
    {
        imgui_init();
        update_swapchain();
        load_fonts();
    }
    ~vulkan_window()
    {
        terminate_on_error(device->waitIdle());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    template <typename Fn> void render(Fn &&fn)
    {
        if (rebuild_swapchain) {
            update_swapchain();
            rebuild_swapchain = false;
        }
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        std::forward<Fn>(fn)();

        ImGui::Render();
        ImDrawData *main_draw_data   = ImGui::GetDrawData();
        const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f ||
                                        main_draw_data->DisplaySize.y <= 0.0f);
        if (!main_is_minimized) {
            auto &frame   = frames[index++ % swapchain_info.minImageCount];
            auto expected = frame.sync.acquire(*swapchain);
            if (expected.result == vk::Result::eErrorOutOfDateKHR ||
                expected.result == vk::Result::eSuboptimalKHR) {
                rebuild_swapchain = true;
            } else {
                assert(expected.result == vk::Result::eSuccess);
                vk::ClearValue clear_value{
                    .color = vk::ClearColorValue{
                        .float32 = {{0.45f, 0.55f, 0.60f, 1.0f}}}};
                frame.sync.render(
                    vk::RenderPassBeginInfo{
                        .renderPass  = *render_pass,
                        .framebuffer = *frames[expected.value].frame_buffer,
                        .renderArea =
                            vk::Rect2D{.extent = swapchain_info.imageExtent},
                        .clearValueCount = 1,
                        .pClearValues    = &clear_value},
                    [main_draw_data](auto command_buffer) {
                        ImGui_ImplVulkan_RenderDrawData(main_draw_data,
                                                        command_buffer);
                    });
                if (auto result =
                        frame.sync.present(*swapchain, expected.value);
                    result == vk::Result::eErrorOutOfDateKHR ||
                    result == vk::Result::eSuboptimalKHR) {
                    rebuild_swapchain = true;
                }
            }
            if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
                ImGui::UpdatePlatformWindows();
                ImGui::RenderPlatformWindowsDefault();
            }
        }
    }
    operator GLFWwindow *() const { return win.get(); }
    operator VkSurfaceKHR() const { return *surface; }
    operator ImGui_ImplVulkan_InitInfo() const { return imgui_vulkan_init; }

private:
    void imgui_init()
    {
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |=
            ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      //
        // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigFlags |=
            ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport /
                                              // Platform Windows
        // io.ConfigViewportsNoAutoMerge = true;
        // io.ConfigViewportsNoTaskBarIcon = true;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        // ImGui::StyleColorsClassic();

        // When viewports are enabled we tweak WindowRounding/WindowBg so
        // platform windows can look identical to regular ones.
        ImGuiStyle &style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            style.WindowRounding              = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForVulkan(*this, true);
        imgui_vulkan_init.Device = *device;
        imgui_vulkan_init.Queue =
            device->getQueue(imgui_vulkan_init.QueueFamily, 0);
        imgui_vulkan_init.PipelineCache   = VK_NULL_HANDLE;
        imgui_vulkan_init.DescriptorPool  = *pool;
        imgui_vulkan_init.Subpass         = 0;
        imgui_vulkan_init.MinImageCount   = swapchain_info.minImageCount;
        imgui_vulkan_init.ImageCount      = swapchain_info.minImageCount;
        imgui_vulkan_init.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
        imgui_vulkan_init.Allocator       = VK_NULL_HANDLE;
        imgui_vulkan_init.CheckVkResultFn = nullptr;
        ImGui_ImplVulkan_Init(&imgui_vulkan_init, *render_pass);

        frames.reserve(imgui_vulkan_init.ImageCount);
        for (auto i = 0; i < imgui_vulkan_init.ImageCount; ++i) {
            frames.push_back(
                frame{.sync = vulkan_sync(*device, *command_pool,
                                          imgui_vulkan_init.Queue)});
        }
    }
    void update_swapchain()
    {
        int width;
        int height;
        glfwGetFramebufferSize(win.get(), &width, &height);
        if (width > 0 && height > 0) {
            terminate_on_error(device->waitIdle());
            auto extent = vk::Extent2D{.width  = static_cast<uint32_t>(width),
                                       .height = static_cast<uint32_t>(height)};
            swapchain   = device->createSwapchainKHRUnique(
                  swapchain_info.setSurface(*surface)
                      .setImageExtent(extent)
                      .setOldSwapchain(*swapchain));
            auto images =
                value_or_terminate(device->getSwapchainImagesKHR(*swapchain));
            for (auto i = 0; i < images.size(); ++i) {
                frames[i].image_view = vulkan_image_view(
                    *device, images[i], swapchain_info.imageFormat);
                frames[i].frame_buffer = vulkan_frame_buffer(
                    *device, *render_pass, *frames[i].image_view, extent);
            }
        }
    }
    void load_fonts()
    {
        frames[0].sync.command(vk::Fence(),
                               ImGui_ImplVulkan_CreateFontsTexture);
        terminate_on_error(device->waitIdle());
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }
};
} // namespace glfw
