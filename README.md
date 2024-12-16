# Insulgo - Your Smart Diabetes Risk Assessment Tool

## Overview
Insulgo is an intelligent web application designed to help users assess their risk of diabetes through advanced machine learning algorithms. Named after "Insulin" and "Go", Insulgo provides a user-friendly, accessible way to get preliminary insights about diabetes risk factors.

## Key Features
- 🔍 Smart Risk Assessment: Advanced machine learning model for accurate diabetes risk prediction
- 📊 Interactive Dashboard: Easy-to-understand visualization of your health metrics
- 🔐 Privacy-First: All data processing happens locally, ensuring your health information stays private
- 📱 Responsive Design: Seamless experience across desktop, tablet, and mobile devices
- 📈 Instant Results: Get your risk assessment results in real-time
- 🎯 Actionable Insights: Receive personalized recommendations based on your results

## How It Works
1. **Input Your Health Data**: Enter basic health metrics like age, BMI, blood pressure, etc.
2. **Instant Processing**: Our machine learning model analyzes your data in real-time
3. **Get Results**: Receive your risk assessment along with personalized insights
4. **Take Action**: Get recommendations for next steps and lifestyle modifications

## 📸 Screenshots
Here's a visual tour of Insulgo:

### Landing Page
![Landing Page](static/img/screenshots/landing.png)
*Welcome to Insulgo - Your diabetes risk assessment companion*

### Risk Assessment Form
![Risk Assessment Form](static/img/screenshots/assessment-form.png)
*Easy-to-use form for entering your health metrics*

### Results Dashboard
![Results Dashboard](static/img/screenshots/results.png)
*Clear visualization of your risk assessment results*

## 🎥 Demo
Watch Insulgo in action:

https://user-images.githubusercontent.com/assets/demo/demo.mp4

<video width="100%" controls>
  <source src="assets/demo/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Note: If the video doesn't play above, you can find it in the `assets/demo` folder of the repository.*

### Quick Feature Tour
1. 🏠 Landing page walkthrough
2. 📝 Filling out the assessment form
3. 📊 Viewing your results
4. 📈 Understanding the recommendations

## Technology Stack
- **Frontend**:
  - HTML5 & CSS3 for structure and styling
  - Modern JavaScript for interactive features
  - Responsive design with custom CSS
- **Backend**:
  - Python for server-side processing
  - Flask web framework
  - Scikit-learn for machine learning capabilities
- **Data Processing**:
  - NumPy and Pandas for data manipulation
  - Custom data generation scripts for model training

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Insulgo.git
cd Insulgo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure
```
Insulgo/
├── static/
│   ├── css/
│   │   └── custom.css      # Custom styling
│   └── img/
│       ├── hero-illustration.svg
│       └── pattern.svg
├── templates/
│   └── predict.html        # Main prediction interface
├── generate_diabetes_data.py
└── README.md
```

## Contributing
We welcome contributions to Insulgo! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer
⚠️ Insulgo is designed as a preliminary risk assessment tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers regarding any medical conditions.

## Support
If you encounter any issues or have questions, please:
1. Check the [Issues](https://github.com/yourusername/Insulgo/issues) page
2. Create a new issue if your problem isn't already listed
3. Provide as much detail as possible about your problem

---

Made with ❤️ for better health awareness
