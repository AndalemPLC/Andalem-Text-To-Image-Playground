from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
import os
import streamlit as st
import torch

def __init__():
 
    global available_gpus
    global device
    global pipeline_choices

    available_gpus = torch.cuda.device_count()

    if available_gpus == 0:

        device = 'cpu'

    else:

        device = 'cuda'

    pipeline_choices = {AutoPipelineForText2Image: 'AutoPipelineForText2Image', 
                        StableDiffusionXLPipeline: 'StableDiffusionXLPipeline'}    

def initialize_page():

    st.set_page_config(page_title = 'Andalem Text-To-Image Playground', 
                       page_icon = './app_images/andalem-icon-orange.png', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')
    
    custom_style = """
                        <style>

                            footer {visibility: hidden;}
                            header {visibility: visible;}
                            #MainMenu {visibility: hidden;}

                            [data-testid=ScrollToBottomContainer] {
                                margin-top: -20px;
                            }

                            [data-testid=stSidebarUserContent] {
                                margin-top: -50px;
                            }
                            
                            [data-testid=stImage] {                                
                                text-align: center;
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                width: 100%;
                            }

                            [data-testid=stImageCaption] {
                                color: #FFFFFF;
                                text-align: center;
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                width: 100%;
                            }

                            [data-testid=StyledFullScreenButton] {
                                display: none;
                            }

                        </style>
                   """
    st.markdown(custom_style, unsafe_allow_html = True)
    
    global prompt
    global negative_prompt
    global number_of_inference_steps
    global guidance_scale
    global image_height
    global image_width
    global chosen_pipeline
    global image_file_name

    st.image('./app_images/andalem-logo.png', width = 245)

    if 'generate_image_button' in st.session_state and st.session_state.generate_image_button == True:

        st.session_state.generating_image = True

    else:

        st.session_state.generating_image = False

    with st.expander('Settings', expanded = True):
    
        prompt = st.text_area('Prompt:', 
                              'A cinematic image of a Samurai Warrior playing basketball.', 
                              height = 50, 
                              key = 'prompt', 
                              disabled = st.session_state.generating_image)
        
        negative_prompt = st.text_area('Negative Prompt:', 
                                       'Lowres, Bad Anatomy, Bad Hands, Cropped, Worst Quality', 
                                       height = 50, 
                                       key = 'negative_prompt', 
                                       disabled = st.session_state.generating_image)

        left_column, right_column = st.columns(2)

        with left_column:

            number_of_inference_steps = st.slider('Number of Inference Steps:', min_value = 1,
                                                                                max_value = 50, 
                                                                                value = 1, 
                                                                                step = 1,
                                                                                key = 'number_of_inference_steps',
                                                                                disabled = st.session_state.generating_image)
            
            image_height = st.slider('Image Height in Pixels:', min_value = 600,
                                                                max_value = 1280, 
                                                                value = 600, 
                                                                step = 8,
                                                                key = 'image_height',
                                                                disabled = st.session_state.generating_image)

            chosen_pipeline = st.selectbox('Pipeline:', 
                                           options = list(pipeline_choices.keys()),
                                           index = 0,
                                           format_func = get_chosen_pipeline,
                                           key = 'pipeline',
                                           disabled = st.session_state.generating_image)
            
        with right_column:

            guidance_scale = st.slider('Guidance Scale:', min_value = 1.0,
                                                          max_value = 20.0, 
                                                          value = 1.0, 
                                                          step = 0.5,
                                                          key = 'guidance_scale',
                                                          disabled = st.session_state.generating_image) 

            image_width = st.slider('Image Width in Pixels:', min_value = 600,
                                                              max_value = 1280, 
                                                              value = 1200, 
                                                              step = 8,
                                                              key = 'image_width',
                                                              disabled = st.session_state.generating_image)
            
            image_file_name = st.text_input('Image File Name:', 
                                            'baller_samurai', 
                                            key = 'image_file_name', 
                                            disabled = st.session_state.generating_image)
            
        generate_image_button = st.button('Generate Image', key = 'generate_image_button', disabled = st.session_state.generating_image)

    if generate_image_button:

        generate_image()
 
def generate_image():

    with st.spinner(text = 'Generating image . . .'):
    
        if available_gpus == 0:


            pipe = chosen_pipeline.from_pretrained('stabilityai/sdxl-turbo', torch_dtype = torch.float32, use_safetensors = True)

        else:

            pipe = chosen_pipeline.from_pretrained('stabilityai/sdxl-turbo', torch_dtype = torch.float16, variant = 'fp16', use_safetensors = True)

        pipe = pipe.to(device)

        generated_image = pipe(prompt = prompt, 
                               negative_prompt = negative_prompt,
                               num_inference_steps = number_of_inference_steps, 
                               guidance_scale = guidance_scale, 
                               height = image_height, 
                               width = image_width).images[0]

    corrected_file_name = image_file_name.lower().replace(' ', '_')
    corrected_file_name = corrected_file_name.replace('(', '')
    corrected_file_name = corrected_file_name.replace(')', '')
    corrected_file_name = corrected_file_name.replace('.', '_')
    corrected_file_name = corrected_file_name.replace(',', '_')
    
    generated_image.save(os.path.join('./generated_images', str(corrected_file_name) + '.png'))

    st.success('Image generated and saved!')
    
    st.image(generated_image, caption = prompt, width = image_width)

def get_chosen_pipeline(chosen_pipeline):

    return pipeline_choices[chosen_pipeline]     

if __name__ == '__main__':

    __init__()

    initialize_page()