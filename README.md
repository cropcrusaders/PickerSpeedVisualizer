# PickerSpeedVisualizer

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Cloning the Repository](#cloning-the-repository)
4. [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
5. [Installing Dependencies](#installing-dependencies)
6. [Running the Application](#running-the-application)
7. [Accessing the Application](#accessing-the-application)
8. [Usage](#usage)
9. [Troubleshooting](#troubleshooting)
10. [Additional Notes](#additional-notes)
11. [Contributing](#contributing)
12. [License](#license)

---

## Introduction

**PickerSpeedVisualizer** is a Dash-based web application designed to visualize and analyze the relationship between picker speed and bale yield in agricultural settings. It provides interactive charts and predictive analytics to help optimize harvesting operations.

---

## Prerequisites

Before you begin, ensure that you have the following software installed on your computer:

- **Python 3.8+**: [Download and install](https://www.python.org/downloads/) from the official Python website.
- **Git**: [Download and install](https://git-scm.com/downloads) from the official Git website.
- **pip**: This usually comes with Python. You can check by running `pip --version` in your terminal.
- **Virtual Environment Module**: Typically included with Python 3 (`venv` module).

---

## Cloning the Repository

Open your terminal or command prompt and navigate to the directory where you want to clone the repository.


# Navigate to your desired directory
cd /path/to/your/directory

# Clone the repository using Git
git clone https://github.com/cropcrusaders/PickerSpeedVisualizer.git

This will create a directory named PickerSpeedVisualizer containing the project files.
Setting Up the Virtual Environment

It's a good practice to use a virtual environment to manage project-specific dependencies.
On Windows:

bash

# Navigate into the project directory
cd PickerSpeedVisualizer

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

On macOS/Linux:

bash

# Navigate into the project directory
cd PickerSpeedVisualizer

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

Once activated, your command prompt should reflect that you're inside the virtual environment (usually by prefixing (venv)).
Installing Dependencies

With the virtual environment activated, install the required Python packages using the requirements.txt file provided in the repository.

bash

pip install -r requirements.txt

This command reads the requirements.txt file and installs all the listed packages into your virtual environment.
Running the Application

Now, you're ready to run the application.

bash

python app.py

This command starts the Dash application server.
Accessing the Application

Once the application is running, you should see output similar to:

csharp

Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: on

Open a web browser and navigate to http://127.0.0.1:8050/ to access the PickerSpeedVisualizer dashboard.
Usage
Adding Your Data

The application uses a CSV file named picker_data.csv or user_data.csv to load data for visualization and analysis. You can use your own data by placing a CSV file with the appropriate structure in the project directory.
CSV File Structure

The CSV file should contain the following columns:

    Date (optional): Format YYYY-MM-DD HH:MM:SS
    PickerSpeed: Speed of the picker machine in km/h
    BalesPerHectare: Yield in bales per hectare
    MaxBaleEjectionSpeed: Maximum speed of bale ejection in km/h
    MaxWrapSpeed or MaxWrapEjectionSpeed: Maximum wrap speed in km/h

Example CSV Content

csv

Date,PickerSpeed,BalesPerHectare,MaxWrapEjectionSpeed
2023-09-22 08:00:00,5.0,3.5,6.5
2023-09-22 09:00:00,5.2,3.7,6.8
2023-09-22 10:00:00,5.5,3.9,7.1

Interacting with the Dashboard

    Sliders: Use the sliders to filter data based on picker speed and bales per hectare.
    Predict Bale Yield: Enter picker speed and max wrap/ejection speed to predict the yield.
    Add New Data: Submit new data points through the form to update the analysis.

#Troubleshooting
Common Issues and Solutions
1. ModuleNotFoundError

Problem: You might encounter an error like ModuleNotFoundError: No module named 'dash'.

Solution: Ensure that you've activated the virtual environment and installed all dependencies.

bash

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

2. Python Version Issues

Problem: The application requires Python 3.8 or higher.

Solution: Check your Python version:

bash

python --version

If it's lower than 3.8, download and install the latest version from the Python website.
3. Port Already in Use

Problem: You receive an error that port 8050 is already in use.

Solution: Specify a different port when running the app:

bash

python app.py --port 8051

Then access the app at http://127.0.0.1:8051/.
4. Permission Denied Errors

Problem: You get permission errors when running commands.

Solution: Try running your terminal or command prompt as an administrator or use sudo on macOS/Linux:

bash

sudo pip install -r requirements.txt

5. No Scatter Plot Displayed

Problem: The scatter plot is not showing up in the application.

Solution:

    Check Data Loading: Ensure your CSV file is correctly named (picker_data.csv or user_data.csv) and placed in the same directory as app.py.

    Verify CSV Content: Ensure the CSV file contains data and the column headers match the expected names.

    Adjust Slider Ranges: The sliders might be filtering out all data. Adjust the sliders to cover the full data range.

    Check Console for Errors: Look for any error messages in the terminal or browser console that could indicate issues.

    Update Dependencies: Ensure all packages are up to date:

    bash

    pip install --upgrade dash plotly pandas

Additional Notes
Updating the Repository

To update your local copy with the latest changes from the repository:

bash

# Navigate to the project directory
cd PickerSpeedVisualizer

# Pull the latest changes
git pull

Deactivating the Virtual Environment

When you're done working with the project:

bash

# Deactivate the virtual environment
deactivate

Running the Application in the Background

If you want the application to keep running after closing the terminal:
On macOS/Linux:

bash

nohup python app.py &

---

On Windows:

Consider using a process manager like pm2 or running the script as a background process using PowerShell.
Customizing the Application

    Data Files: Replace picker_data.csv with your own data, ensuring it has the same structure.

    Model Files: If you have a pre-trained model, replace reg_model.pkl with your model file.

    Debug Mode: By default, app.py runs in debug mode. For production, set debug=False:

    python

    if __name__ == '__main__':
        app.run_server(debug=False)

Installing Git and Python

If you don't have Git or Python installed:
Installing Git:

    Windows: Download and install from Git for Windows.
    macOS: Install via Homebrew (brew install git) or download from Git SCM.
    Linux: Use your distribution's package manager, e.g., sudo apt-get install git for Debian-based systems.

Installing Python:

    Windows and macOS: Download the latest Python 3 installer from the official website.
    Linux: Python 3 is usually pre-installed. Check with python3 --version. If not, install it via your package manager.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
License

This project is licensed under the MIT License.
Conclusion

You now have the PickerSpeedVisualizer application installed and running on your PC. You can interact with the dashboard, explore the data visualizations, and utilize the predictive analytics feature.
Contact

For any questions or support, please open an issue in the GitHub repository.

