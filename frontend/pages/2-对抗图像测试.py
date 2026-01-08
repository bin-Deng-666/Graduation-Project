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
    
    .top-section {
        display: flex;
        gap: 20px;
        margin-bottom: 32px;
        align-items: center;
        padding: 24px;
        background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .image-id-input {
        flex: 1;
        max-width: 500px;
        padding: 16px 24px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        background: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        font-family: inherit;
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
    
    .image-container, .answer-container {
        flex: 1;
        min-width: 380px;
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    .image-container h3, .answer-container h3 {
        margin-bottom: 16px;
        color: #1a1a2e;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        position: relative;
        padding-bottom: 8px;
    }
    
    .image-container h3::after, .answer-container h3::after {
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
    
    .image-wrapper {
        width: 100%;
        height: 420px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 8px;
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
    
    .prompt-input {
        flex: 1;
        padding: 16px 20px;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-size: 16px;
        resize: vertical;
        min-height: 100px;
        font-family: inherit;
        background: #f8fafc;
        line-height: 1.6;
    }
    
    .prompt-input:focus {
        outline: none;
        border-color: #4895ef;
        box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.1);
        background: white;
    }
    
    .answer-card {
        padding: 20px;
        min-height: 220px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
        line-height: 1.7;
        color: #4a5568;
        word-wrap: break-word;
        overflow-y: auto;
        font-size: 15px;
    }
    
    .placeholder-answer {
        width: 100%;
        min-height: 220px;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #4a5568;
        font-size: 1.1rem;
        border-radius: 8px;
        padding: 20px;
        border: 2px dashed #cbd5e0;
    }

    .main-title {
        text-align: center;
        margin: 40px 0 20px 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# 创建页面标题区域
def render_page_header():
    st.markdown('''
    <div class="page-header">
        <h1 class="page-title">对抗图像测试</h1>
        <p class="page-subtitle">测试对抗图像的效果，对比原始图像和对抗图像的模型预测结果</p>
    </div>
    ''', unsafe_allow_html=True)

# 初始化会话状态
if 'original_image_url' not in st.session_state:
    st.session_state.original_image_url = None

if 'adversarial_image_url' not in st.session_state:
    st.session_state.adversarial_image_url = None

if 'original_image_answer' not in st.session_state:
    st.session_state.original_image_answer = None

if 'adversarial_image_answer' not in st.session_state:
    st.session_state.adversarial_image_answer = None

if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

# 加载图像函数
def load_images(image_id_value):
    if not image_id_value:
        st.warning('请输入图像ID')
        return

    st.session_state.is_loading = True
    
    try:
        # TODO: 实现与后端API的交互逻辑
        # 1. 根据imageId从后端获取原始图像和对抗图像的URL
        # 2. 更新originalImageUrl和adversarialImageUrl
        # 3. 清空之前的回答
        
        # 模拟API调用延迟
        time.sleep(1.5)
        
        # 示例API调用格式:
        # import requests
        # response = requests.get(f'/api/images/{image_id_value}')
        # data = response.json()
        # st.session_state.original_image_url = data['originalImageUrl']
        # st.session_state.adversarial_image_url = data['adversarialImageUrl']
        
        # 清空之前的回答
        st.session_state.original_image_answer = None
        st.session_state.adversarial_image_answer = None
        
        # 这里可以添加示例图像URL用于测试
        # st.session_state.original_image_url = 'https://via.placeholder.com/600x400?text=Original+Image'
        # st.session_state.adversarial_image_url = 'https://via.placeholder.com/600x400?text=Adversarial+Image'
        
    except Exception as error:
        st.error(f'加载图像失败: {error}')
    finally:
        st.session_state.is_loading = False

# 获取AI回答函数
def get_answers(image_id_value, prompt_value):
    if not image_id_value or not prompt_value:
        st.warning('请输入图像ID和提示文本')
        return

    st.session_state.is_loading = True
    
    try:
        # TODO: 实现与后端API的交互逻辑
        # 1. 发送imageId和prompt到后端API
        # 2. 获取原始图像和对抗图像的模型回答
        # 3. 更新originalImageAnswer和adversarialImageAnswer
        
        # 模拟API调用延迟
        time.sleep(2)
        
        # 示例API调用格式:
        # import requests
        # response = requests.post('/api/answers', json={
        #     'imageId': image_id_value,
        #     'prompt': prompt_value
        # })
        # data = response.json()
        # st.session_state.original_image_answer = data['originalAnswer']
        # st.session_state.adversarial_image_answer = data['adversarialAnswer']
        
        # 这里可以添加示例回答用于测试
        # st.session_state.original_image_answer = f"原始图像回答：{prompt_value} (Image ID: {image_id_value})"
        # st.session_state.adversarial_image_answer = f"对抗图像回答：{prompt_value} (Image ID: {image_id_value})"
        
    except Exception as error:
        st.error(f'获取回答失败: {error}')
    finally:
        st.session_state.is_loading = False

# 渲染页面标题
render_page_header()

# 主容器
st.markdown('<div class="container">', unsafe_allow_html=True)

# 图像ID输入和加载按钮同一行
st.markdown('<h2 class="main-title">图像输入</h2>', unsafe_allow_html=True)
col_id, col_btn = st.columns([4, 1])
with col_id:
    image_id = st.text_input("图像ID", placeholder="请输入图像ID", key="image_id_input", help="输入要加载的图像ID")
with col_btn:
    st.markdown('<div class="button-column">', unsafe_allow_html=True)
    load_button = st.button("加载图像", key="load_button", help="点击加载图像", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 当点击加载按钮时调用函数
if load_button:
    load_images(image_id)

# 中间：图像对比区域 - 使用Streamlit原生双栏布局
image_col1, image_col2 = st.columns(2)

with image_col1:
    st.markdown('<h5 style="text-align: center;">原始图像</h5>', unsafe_allow_html=True)
    
    if st.session_state.original_image_url:
        st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
        st.image(st.session_state.original_image_url, caption="原始图像", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="placeholder-image">原始图像</div>', unsafe_allow_html=True)

with image_col2:
    st.markdown('<h5 style="text-align: center;">对抗图像</h5>', unsafe_allow_html=True)
    
    if st.session_state.adversarial_image_url:
        st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
        st.image(st.session_state.adversarial_image_url, caption="对抗图像", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="placeholder-image">对抗图像</div>', unsafe_allow_html=True)

st.divider()
st.markdown("&nbsp;")

# 底部：Prompt 输入和回答显示
st.markdown('<h2 class="main-title">文本输入</h2>', unsafe_allow_html=True)
# 文本输入和获取回答按钮同一行
col_prompt, col_answer_btn = st.columns([4, 1])
with col_prompt:
    prompt = st.text_area("请输入提示文本", placeholder="请输入提示文本", height=100, key="prompt_input", help="输入要询问的问题")
with col_answer_btn:
    st.markdown("&nbsp;")
    get_answers_button = st.button("获取回答", key="get_answers_button", help="点击获取AI回答", use_container_width=True)

# 当点击获取回答按钮时调用函数
if get_answers_button:
    get_answers(image_id, prompt)

# 回答显示区域 - 使用Streamlit原生双栏布局
answer_col1, answer_col2 = st.columns(2)

with answer_col1:
    st.markdown('<h5 style="text-align: center;">原始图像回答</h5>', unsafe_allow_html=True)

with answer_col2:
    st.markdown('<h5 style="text-align: center;">对抗图像回答</h5>', unsafe_allow_html=True)

# 加载状态指示器
if st.session_state.is_loading:
    with st.spinner('正在加载...'):
        # 这里不需要额外的代码，spinner会自动处理加载状态
        pass

st.markdown('</div>', unsafe_allow_html=True)