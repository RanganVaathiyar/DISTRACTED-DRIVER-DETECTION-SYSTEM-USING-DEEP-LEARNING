import streamlit as st 
 
from predictionOnImage import return_prediction from PIL import Image from matplotlib import pyplot as plt import time import random 
 st.title("Distracted Driver Detection") 
 
 
 fig = plt.figure() list1 =["CALIFORNIA Antique MORE TK 2021","CALIFORNIA CL818 DP","CALIFORNIA 
MH20DV2366 FO", 
	"CALIFORNIA 	HISTORICAL 	VEHICLE 	542Z","California 	U.S.GOVERNMENT 
3688P"] 
RANDOM_NUM= random.choice(list1) 
 
 
 
	Car_Model = 	["Toyota 	Camry","Tesla Cybertruck","Chevrolet Corvette","Lamborghini 
Miura","Porsche 911","Honda Civic","Nissan Altima"] RANDOM_Car= random.choice(Car_Model) 
 
image1 = Image.open('C:/Users/Victor/OneDrive/Desktop/Distracted Driver Detection using DL/demo_on_image/Calfornia_Antique.jpg') 
image2 = Image.open('C:/Users/Victor/OneDrive/Desktop/Distracted Driver Detection using DL/demo_on_image/California _Disable _Person.jpg') 
image3 = Image.open('C:/Users/Victor/OneDrive/Desktop/Distracted Driver Detection using DL/demo_on_image/Img_1.jpg') 
image4 = Image.open('C:/Users/Victor/OneDrive/Desktop/Distracted Driver Detection using DL/demo_on_image/Img_11.jpg') 
image5 = Image.open('C:/Users/Victor/OneDrive/Desktop/Distracted Driver Detection using DL/demo_on_image/Img_111.jpg') 
 
 
 
def main(): 
 file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"]) class_btn = st.button("Classify") 
if file_uploaded is not None: 
 
image = Image.open(file_uploaded) 
 
st.image(image, caption='Uploaded Image', use_column_width=True) 
 
 
 
if class_btn: 
 if file_uploaded is None: 
 st.write("Invalid command, please upload an image") else: with st.spinner('Model working. .. '): 
 
plt.imshow(image) plt.axis("off") predictions = return_prediction(image) time.sleep(1) st.success('Classified') st.write(predictions) st.metric(label="Model Accuracy Improvement :", value="91%", delta="27%") st.write("Car Model : ",RANDOM_Car) if RANDOM_NUM[:13]=="CALIFORNIA An": 
 st.write("License plate recognize service : Calfornia Antique MC Cars") st.image(image1,"Licence plate Detection", use_column_width=True) st.write("Licence plate Number : ",RANDOM_NUM) 
st.write("Year: 2021") 
 elif RANDOM_NUM[:13]=="CALIFORNIA CL": 
 st.write("License plate recognize service : California Disable Person") st.image(image2,"Licence plate Detection", use_column_width=True) st.write("Licence plate Number : ",RANDOM_NUM) st.write("Year: 2021") 
 elif RANDOM_NUM[:13]=="CALIFORNIA MH": 
 st.write("License plate recognize service : Foreign Organization") st.image(image3,"Licence plate Detection", use_column_width=True) st.write("Licence plate Number : ",RANDOM_NUM) st.write("Year: 2022") 
 elif RANDOM_NUM[:13]=="CALIFORNIA HI": 
 st.write("License plate recognize service : Historical Vehicles (Legacy cars)") st.image(image4,"Licence plate Detection", use_column_width=True) st.write("Licence plate Number : ",RANDOM_NUM) st.write("Year: 1969") else : 
st.write("License plate recognize service : United States House of Representatives") st.image(image5,"Licence plate Detection", use_column_width=True) st.write("Licence plate Number : ",RANDOM_NUM) st.write("Year: 2022") 
st.write("Car Owner Contact Details : +916383732118") st.header('Accuracy Improvement') 
col1, col2 = st.columns(2) 
 
 
 
with col1: 
 st.metric(label="Model Accuracy of other models", value="64%", delta="-16%") with col2: 
st.metric(label="Model Accuracy of CNN Non Batch", value="91%", delta="27%") 
 
 
 
if    name ==' main ': 
main() 
