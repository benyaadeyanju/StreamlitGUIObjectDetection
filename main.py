from time import time
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO  = 'demo.mp4'


st.title('Face Mesh App Using MediaPipe')

st.markdown(
      """
      <style>
      [data-testid="stSidebar"][aria=expanded="true"] > div:first-child{
          width: 350px
      }
      [data-testid="stSidebar"][aria=expanded="false"] > div:first-child{
          width: 350px;
          margin-left : -350px
      }
      </style>
      """,

      unsafe_allow_html = True,
)

st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('Parameters')


@st.cache()
def image_resize(image,width=None,height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    #resize the image
    resize = cv2.resize(image, dim,interpolation=inter)
    
    return resize

app_mode = st.sidebar.selectbox('Choose the App mode',
                               ['About App','Run on Image','Run on Video']
                               )

if app_mode == 'About App':
    st.markdown('In this Application we are using **MediaPipe** for creating a FaceMesh App and  **streamlit** is used to create a Graphical User Interface(GUI) Application ')
    st.markdown(
      """
      <style>
      [data-testid="stSidebar"][aria=expanded="true"] > div:first-child{
          width: 350px
      }
      [data-testid="stSidebar"][aria=expanded="false"] > div:first-child{
          width: 350px;
          margin-left : -350px
      }
      </style>
      """,

      unsafe_allow_html = True,
)
    st.video('https://youtu.be/JzHNIcvpGk8')

    st.markdown('''

           # About Me\n

             Hey this is **Benya Adeyanju Jamiu**  PhD student  from **Multimedia University,Melaka**. \n

             If you are intrested in studying Artificial Intelligence and Computer Vision, you can contact 
             us at our school website. \n

             If you need more information about the program you can reach me through my **E-mail / Social Media PlatForm**. \n

             - [Melaka Campus](https://www.mmu.edu.my/book-an-appointment/?gclid=Cj0KCQjwsdiTBhD5ARIsAIpW8CLxlhZ0NFb9JFsTR-lMcFLjG7qoqr0jadphFyuXR9k4pGA6NuiRBpsaAt1WEALw_wcB)
             - [FaceBook](https://www.facebook.com/mmumalaysia/)
             - [YouTube](https://www.youtube.com/user/mmumalaysiatv)
             - [Linkedin](https://www.linkedin.com/school/mmumalaysia/mycompany/)
             - [Twitter](https://twitter.com/mmumalaysia)


             ''') 
 
##-------------------------------------------Run on Image-----------------------------------
elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    

    st.sidebar.markdown('---')

    st.markdown(
      """
      <style>
      [data-testid="stSidebar"][aria=expanded="true"] > div:first-child{
          width: 350px
      }
      [data-testid="stSidebar"][aria=expanded="false"] > div:first-child{
          width: 350px;
          margin-left : -350px
      }
      </style>
      """,

      unsafe_allow_html = True,

    )

    st.markdown("**Detected Faces**")
    kp1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value =1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value= 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Orginal Image')
    st.sidebar.image(image)

    face_count = 0

    ##Dashboard 

    with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = max_faces,
    min_detection_confidence = detection_confidence) as face_mesh:
        
        results = face_mesh.process(image)
        out_image = image.copy()

         ##Face LandMark Drawing
        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)
            face_count += 1

            mp_drawing.draw_landmarks(
            image = out_image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            #connections = mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec = drawing_spec)

    
            kp1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>",unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image,use_column_width=True)


#----------------------------On Video-------------------------------------------------------

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding',False)

    use_webcam = st.sidebar.button('use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value=True)

    #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    #st.sidebar.markdown('---')

    st.markdown(
      """
      <style>
      [data-testid="stSidebar"][aria=expanded="true"] > div:first-child{
          width: 350px
      }
      [data-testid="stSidebar"][aria=expanded="false"] > div:first-child{
          width: 350px;
          margin-left : -350px
      }
      </style>
      """,

      unsafe_allow_html = True,

    )

    #st.markdown("**Detected Faces**")
    #kp1_text = st.markdown("0")

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value =1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value= 0.5)
    tracking_confidence =  st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value= 0.5)
    st.sidebar.markdown('---')

    st.markdown("## Output")

    stframe = st.empty
    video_file_buffer = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    

    ## We get out input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)


    # width = vid.get(cv2.CAP_PROP_FRAME_WIDTH )
    # height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT )
    # fps =  vid.get(cv2.CAP_PROP_FPS)

    width = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    height = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    fps_input = vid.set(cv2.PROP_FRAME_FPS)
    #width = int(vid.get(cv2.CAP_PROR_FRAME_WIDTH))
    #height = int(vid.get(cv2.CAP_PROR_FRAME_HEIGHT))
    #fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording part
    #codec = cv2.VideoWriter_fourcc('M','J','P','G')
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input(width,height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)



    # face_count = 0

    # ##Dashboard 

    # with mp_face_mesh.FaceMesh(
    # static_image_mode = True,
    # max_num_faces = max_faces,
    # min_detection_confidence = detection_confidence) as face_mesh:
        
    #     results = face_mesh.process(image)
    #     out_image = image.copy()

    #      ##Face LandMark Drawing
    #     for face_landmarks in results.multi_face_landmarks:
    #         print('face_landmarks:', face_landmarks)
    #         face_count += 1

    #         mp_drawing.draw_landmarks(
    #         image = out_image,
    #         landmark_list = face_landmarks,
    #         connections = mp_face_mesh.FACEMESH_CONTOURS,
    #         #connections = mp_face_mesh.FACE_CONNECTIONS,
    #         landmark_drawing_spec = drawing_spec)

    
    #         kp1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>",unsafe_allow_html=True)
    #     st.subheader('Output Image')
    #     st.image(out_image,use_column_width=True)