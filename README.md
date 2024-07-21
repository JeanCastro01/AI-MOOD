# AI Mood Project

AI Mood is the capstone project for my Information Technology degree, designed to leverage advanced technologies for real-time facial emotion recognition and analysis. This project was developed using the Django framework for a robust web application structure, and machine learning techniques were implemented using scikit-learn (sklearn). The core of our emotion recognition system is built on Convolutional Neural Networks (CNN), which are highly effective in image processing and pattern recognition.

## Functionality Overview

- **User Authentication:** Upon logging in, users gain access to the main dashboard, which serves as the control center for the application.
- **Emotion Capture:** By pressing the "Start Recording" button, the camera is activated to capture facial expressions. The system is capable of recognizing and recording up to seven distinct emotions from anyone who appears in front of the camera. The capturing process is designed to be seamless and user-friendly.
- **Data Storage:** After capturing the desired facial expressions, users return to the main dashboard and select the "Stop Recording" button. This action stores all captured emotions in a database, ensuring that the data is well-organized and retrievable for further analysis.
- **Emotion Analysis:** The "Chart" page is where users can view detailed analyses of the emotions captured by the system. The trained CNN model processes the stored facial expression data to determine the emotional states. This analysis is then presented through various charts and graphs, providing users with clear and insightful visual representations of the emotional data.

## Technical Details

- **Framework:** Django was chosen for its scalability and ease of integration with various components required for this project.
- **Machine Learning:** Scikit-learn (sklearn) provided the necessary tools for training our machine learning models. The choice of a Convolutional Neural Network (CNN) was due to its superior performance in image recognition tasks.
- **Database:** The captured emotional data is stored in a structured database, facilitating efficient data management and retrieval for analysis.
- **User Interface:** The application features an intuitive user interface, ensuring that users can easily navigate through the different functionalities, from recording emotions to viewing detailed analyses.

AI Mood not only showcases my technical skills in software development and machine learning but also highlights my ability to create practical applications that can provide real-world benefits. This project represents a significant step in the integration of AI with everyday technology, aiming to offer a unique perspective on emotion recognition and analysis.

---

Feel free to contribute to this project or reach out for collaborations and inquiries!
