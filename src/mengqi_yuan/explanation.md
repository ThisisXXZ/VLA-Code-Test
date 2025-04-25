## Explanation

* **Main Task**: I designed a task where multiple cups of different heights are set up, with each cup randomly positioned either upright or sideways. The robot's task is to correct the orientation of all the cups and place them in a designated area from left to right in ascending order of height.  

* **Core Design**:  
  * During initialization, cups can be either upright or sideways, but they must not be embedded into each other. Their positioning should reflect real-world physical constraints.  
  * The final arrangement requires all cups to be in an upright position and correctly ordered by height in the specific area