# Object Detection & Analysis in Football Match Videos

## Problem description
This project aims to develop a system for detecting key objects in football match recordings, applying computer vision techniques to the sports domain, specifically football. The primary objective is to build a robust and efficient pipeline for object detection in football match videos.

Two object detection models will be utilized. The first model detects high-level objects such as football players, goalkeepers, referees, and the ball. The second model is designed specifically to focus on the football players. It will perform nested object detection within the frames of the detected players to identify the jersey numbers on their shirts, enabling the tracking of individual players and extraction of key information valuable to coaches and football analysts.

By tracking multiple parameters related to these detected objects, a detailed analysis of the game—or at least certain parts of it—can be generated. A step further would be fine-tuning a large language model (LLM) to perform basic reasoning about the game using metadata from object tracking. With the inclusion of a temporal component, the LLM could potentially generate live commentary for the football match.

## Datasets
The biggest source of object detection datasets, including ones related to football, can be found at [Roboflow](https://roboflow.com/), an open source platform with dozens of annotated datasets.
<li>High-level object detection dataset</li>
<blockquote> 
  A dataset for high-level objects detection that gave the best results so far is given at the following <a href="https://universe.roboflow.com/meriem-ahjouji/mini_projet_football">link</a>. It consists of 9814 annotated images of a football match, with a variety of different recording angles and perspectives covered.
</blockquote>

<li>Jersey number detection dataset</li>
<blockquote>
  When it comes to jersey number detection, the best results so far were obtained by fusing two datasets into one (<a href="https://universe.roboflow.com/pusan-national-university-aajlj/jersey-number-detection-8a55j">link1</a>, <a href="https://universe.roboflow.com/volleyai-actions/jersey-number-detection-s01j4">link2</a>). As it can be seen from the second link, the second dataset is not even related to football, but to the volleyball. In other words, when it comes to jersey number detection, the sport being played is not even important at that point, only the features related to jersey number itself and to the variation of angles, fonts and perspectives at which the images were taken. 
</blockquote>

## Models
It is important to mention that this project implemented two approaches to the object detection so far. 
<ol>
  <li>YOLO Darknet & VGG16 - The initial approach used YOLO Darknet framework for high-level object detection and the VGG16 model for jersey number classification. Training was performed both from scratch and by fine-tuning pre-existing models. However, the results obtained were not nearly good enough to keep on experimenting with this approach, especially when it comes to number classification using VGG. </li>
  <li>YOLOv5 - The second approach utilized YOLOv5 for both high-level and low-level detections. With approximately 97 million parameters, YOLOv5 is a larger and more complex model than YOLO Darknet or VGG16, therefore it significantly outperformed the initial approach. Currently, the high-level detection works nearly flawlessly, although there is still some room for improvement. The number classification on tracklets has yet to be evaluated, although it performed reasonably well on images.</li>
</ol>

## Results
By far representative results in high-level object detection are summarized in the following video.
