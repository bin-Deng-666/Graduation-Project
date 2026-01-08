import streamlit as st
import time

# 创建自定义CSS样式
st.markdown("""
<style>
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 24px 40px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* 页面标题区域 */
    .page-header {
        background: white;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        margin-bottom: 30px;
        padding: 30px;
        text-align: center;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 600;
        color: #4361ee;
        margin-bottom: 10px;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #4a5568;
    }
    
    .image-container {
        flex: 1;
        min-width: 380px;
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    .image-container h3 {
        margin-bottom: 16px;
        color: #1a1a2e;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        position: relative;
        padding-bottom: 8px;
    }
    
    .main-title {
        text-align: center;
        margin: 40px 0 20px 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    
    .image-container h3::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, #4895ef, #f72585);
        border-radius: 3px;
    }
    
    .placeholder-image {
        width: 100%;
        height: 420px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #4a5568;
        font-size: 1.1rem;
        border-radius: 8px;
        border: 2px dashed #cbd5e0;
    }
    
    .bottom-section {
        display: flex;
        gap: 20px;
        margin-bottom: 32px;
        align-items: flex-end;
        padding: 24px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        width: 100%;
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    .load-button, .get-answer-button {
        padding: 14px 28px;
        background: linear-gradient(135deg, #4895ef 0%, #f72585 100%);
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.2);
    }
    
    /* 确保输入框和按钮底部对齐 */
    .stTextInput, .stButton {
        margin-bottom: 0;
    }
    
    /* 调整按钮容器的垂直对齐 */
    .button-column {
        display: flex;
        align-items: flex-end;
    }
</style>
""", unsafe_allow_html=True)

# 创建页面标题区域
def render_page_header():
    st.markdown('''
    <div class="page-header">
        <h1 class="page-title">对抗图像生成</h1>
        <p class="page-subtitle">输入图像ID加载图像，选择攻击算法，生成对抗样本</p>
    </div>
    ''', unsafe_allow_html=True)

# 初始化会话状态
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

if 'generated_adversarial_image' not in st.session_state:
    st.session_state.generated_adversarial_image = None

if 'loaded_image' not in st.session_state:
    st.session_state.loaded_image = None

if 'image_id' not in st.session_state:
    st.session_state.image_id = None

# 渲染页面标题
render_page_header()

# 主容器
st.markdown('<div class="container">', unsafe_allow_html=True)

# 图像ID输入和原始图像区域
st.markdown('<h2 class="main-title">原始图像</h2>', unsafe_allow_html=True)

# 图像ID输入和加载按钮同一行
col_id, col_btn = st.columns([4, 1])
with col_id:
    image_id = st.text_input("图像ID", placeholder="请输入图像ID", value=st.session_state.image_id)
with col_btn:
    st.markdown('<div class="button-column">', unsafe_allow_html=True)
    load_button = st.button("加载图像", key="load_image_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 加载图像逻辑
if load_button:
    if image_id:
        with st.spinner("正在加载图像..."):
            # 图像加载函数（空实现）
            def load_image_by_id(id):
                # TODO: 实现根据ID加载图像的逻辑
                # 这里应该从数据库或文件系统获取图像
                return None
            
            # 调用加载图像函数
            loaded_image = load_image_by_id(image_id)
            
            # 模拟加载成功
            time.sleep(1)
            st.session_state.image_id = image_id
            st.session_state.loaded_image = loaded_image
            
            st.success(f"成功加载图像 ID: {image_id}")
    else:
        st.warning("请输入图像ID")

# 显示加载的图像
if st.session_state.loaded_image:
    st.image(st.session_state.loaded_image, caption=f"加载的图像 (ID: {st.session_state.image_id})")
else:
    # 加载前为空，不显示任何内容
    pass
st.divider()
st.markdown("&nbsp;")


# 攻击参数配置区域
st.markdown('<h2 class="main-title">攻击参数配置</h2>', unsafe_allow_html=True)

attack_params_col1, attack_params_col2, attack_params_col3 = st.columns(3)

with attack_params_col1:
    attack_method = st.selectbox(
        "攻击方法",
        ["FGSM", "PGD", "CW", "DeepFool", "BIM", "MIM"],
        help="选择对抗攻击算法"
    )

with attack_params_col2:
    epsilon = st.slider(
        "扰动强度 (ε)",
        min_value=0.0,
        max_value=0.5,
        value=0.03,
        step=0.01,
        help="对抗扰动的强度，值越大攻击效果越强"
    )

with attack_params_col3:
    iterations = st.number_input(
        "迭代次数",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="迭代优化的次数，影响攻击成功率和图像质量"
    )

# 模型选择
model_name = st.selectbox(
    "目标模型",
    ["ResNet50", "VGG16", "InceptionV3", "DenseNet121", "自定义模型"],
    help="选择要攻击的目标模型"
)

# 高级选项
with st.expander("高级选项"):
    targeted_attack = st.checkbox("靶向攻击", help="是否指定攻击目标类别")
    if targeted_attack:
        target_class = st.text_input("目标类别", placeholder="例如：cat, dog, car")
    
    confidence = st.slider("攻击置信度", 0.5, 1.0, 0.9, 0.05)
    
    visualization = st.checkbox("生成扰动可视化", value=True)
    if visualization:
        vis_type = st.selectbox("可视化类型", ["差分图像", "扰动热图", "叠加显示"])

# 生成按钮 - 居中显示
col_center = st.columns([1, 1, 1])
with col_center[1]:  # 中间的列用于居中按钮
    gen_adv_button = st.button("生成对抗图像", key="generate_btn", type="primary", use_container_width=True)

if gen_adv_button:
    if not st.session_state.loaded_image:
        st.warning("请先加载原始图像")
    else:
        with st.spinner("正在生成对抗图像..."):
            # 模拟生成过程
            time.sleep(2)
            
            # 这里应该是实际的对抗图像生成代码
            # 示例：st.session_state.generated_adversarial_image = generate_adversarial_image(st.session_state.loaded_image, attack_method, epsilon, iterations)
            st.session_state.generated_adversarial_image = st.session_state.loaded_image  # 临时使用加载的图像作为对抗图像
            
            st.success("对抗图像生成完成！")
            # 可以在这里保存生成的图像
            # st.download_button("下载对抗图像", data=generated_image_data, file_name="adversarial_image.png", mime="image/png")
st.divider()
st.markdown("&nbsp;")

# 对抗图像显示区域
st.markdown('<h2 class="main-title">对抗图像</h2>', unsafe_allow_html=True)

col_center = st.columns([1, 2, 1])
with col_center[1]:  # 中间的列用于居中按钮
    load_adv_button = st.button("加载对抗图像", key="load_adv_btn", use_container_width=True)
if load_adv_button:
    # 这里可以添加加载对抗图像的逻辑
    st.info("加载对抗图像功能待实现")