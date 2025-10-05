# 🚗 Parking Spot Occupancy Detection Using Computer Vision and CNN

## 📘 Overview
This project presents an AI-powered **Parking Spot Occupancy Detection System** that leverages **Computer Vision** and **Convolutional Neural Networks (CNN)** to automatically identify whether parking spots are **occupied or empty** from video footage.  
It eliminates the need for hardware sensors by using only a camera feed, providing a **scalable, cost-effective, and real-time** smart parking solution.

---

## ⚙️ Key Features
- 🎯 Detects occupied and empty parking spots using CNN-based image classification  
- 🧠 Combines **ROI-based change detection** with **deep learning** for optimized performance  
- 🖼️ Real-time video annotation with color-coded bounding boxes and spot counters  
- 💡 Works on any camera feed — no sensors or additional infrastructure required  

---

## 🧩 Methodology
1. **Parking Spot Identification** – Extracts individual Regions of Interest (ROIs) using mask images and connected components.  
2. **Frame Sampling & Change Detection** – Detects frames with visual changes to minimize redundant processing.  
3. **CNN Classification** – Classifies each ROI as *Occupied* or *Empty* using a trained CNN model.  
4. **Visualization** – Displays real-time annotated output with total and available spot counts.

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries & Frameworks:** OpenCV, NumPy, TensorFlow/Keras (or PyTorch), Scikit-learn, Matplotlib  
- **Model:** Convolutional Neural Network (CNN)  
- **Tools:** Jupyter Notebook / Google Colab  

---

## 🧪 Workflow

- Video Input → ROI Extraction → Frame Change Detection → CNN Classification → Visualization → Output

---

## 📊 Results
- Achieved high accuracy in classifying parking occupancy.  
- Reduced computation time through selective frame analysis.  
- Provided clear and real-time visualization of parking lot status.

---

## 🌐 Applications
- Smart parking systems in **malls, airports, campuses, and smart cities**  
- Real-time parking assistance in mobile apps  
- Parking data analytics for **urban planning and traffic management**

---

## 🚀 Future Enhancements
- Deploy on **edge devices** (e.g., Raspberry Pi, Jetson Nano) for real-time inference  
- Integrate with **IoT and cloud dashboards** for centralized monitoring  
- Expand dataset to support **different lighting and weather conditions**





