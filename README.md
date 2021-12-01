# Parametric visualization of your project data for Revit and IFC files
Any medium-sized construction project is a source of big data with hundreds of thousands of different elements, which in turn have tens or hundreds of different parameters or properties. Up from now to properly understand this data you can use the BIMJSON format to visualize all the information on all elements and its properties as a multidimensional point cloud.
### Screenshots
![enter image description here](https://opendatabim.io/wp-content/uploads/2021/12/ezgif-4-01b7e76c8c76.gif)

#  Seeing changes visually

In the first version of the project within the visualization on the left is the sum of volumes for the foundation Bearing Footing – 700×230 – 11 m³. In the second version of the project, the dimensions of the foundations were changed, which affected the total volume – 147 m³ of the position and is displayed in the visualization by an enlarged group symbol. 


![enter image description here](https://opendatabim.io/wp-content/uploads/2021/12/Unbenannt-2.png)
## Find data outliers and insights
Such a visual matrix of projects can be compared to a snapshot of the lung or unique DNA of a particular project or project area.

Having a large base of such images (or DNA), we can by similarity (approximation), as machine algorithms do it from lung images or images of people for lidars, to determine cost or time characteristics for similar parts or whole new projects that will consist of similar pieces of old projects.

![enter image description here](https://opendatabim.io/wp-content/uploads/2021/10/Ein-bisschen-Text-hinzufugen-3.png)


## Built With

-   [Dash](https://dash.plot.ly/)  - Main server and interactive components
-   [Plotly Python](https://plot.ly/python/)  - Used to create the interactive plots


## Requirements

We suggest you to create a separate virtual environment running Python 3 for this app, and install all of the required dependencies there. Run in Terminal/Command Prompt:

```
git clone https://github.com/OpenDataBIM/VisualBIM.git
cd VisualBIM
python3 -m virtualenv venv

```

In UNIX system:

```
source venv/bin/activate

```

In Windows:

```
venv\Scripts\activate

```

To install all of the required packages to this environment, simply run:

```
pip install -r requirements.txt

```

and all of the required  `pip`  packages, will be installed, and the app will be able to run.

## [](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard#how-to-use-this-app)How to use this app

Run this app locally by:

```
python index.py

```

Open  [http://0.0.0.0:3000/](http://0.0.0.0:3000/)  in your browser, you will see a live-updating dashboard.

