# VLA-Code-Test

This coding test is designed for you to showcase your ability to define custom tasks in the Maniskill simulation environment for our VLA project. Your main role will be to design tasks that challenge both digital and physical agent capabilities. The focus is on creating a task that is distinct from existing tasks, with an emphasis on the robotics (physical agent) component, at least for this coding test.

---

## Overview

In this test, you are required to define a custom task that involves interaction between the digital and physical worlds. Although the overall task may include both digital decision-making and physical execution, you are only required to implement the physical (robotics) component. For example, if the digital agent chooses a package based on criteria (like color), your implementation should focus on enabling the robot to pick up a package of that specified color.

---

## Key Details

### Time Limit
- **Deadline:** March 26, 11:59 PM

### Evaluation Criteria
1. **Correctness:**  
   - This is the most critical criterion. Your final version must be executable using our provided `/src/runner.py`.  
   - You may include a custom runner with additional features (such as logging) if desired, though it is not required.

2. **High-Level Task Design:**  
   - Deviation: Your task should be as distinct as possible from existing tasks, from the initial design concept to the actual implementation.
   - Interaction: Your task is encouraged to involve both realms. However, your implementation is required only for the physical agent (robotics) aspect. *Example:* The digital agent selects a package based on criteria (e.g., color chosen from an Amazon account cart), while the physical agent’s task is to "pick up a package with a certain color."
   - Complexity: The overall difficulty, reflecting the challenge for a robot to complete or learn the task through training.

3. **Advanced Features (Bonus):**  
   - Extra credit will be given for incorporating advanced features such as custom agents/objects, physical dynamics, or enhanced observations.

4. **Code Style:**  
   - You should have readable and clean code style, which is critical to team collaboration in big projects. Add necessary comments to help us understand your code. While important, code style is a minor evaluation factor compared to functionality and task design. 

---

## Files Provided

- **Documentation:**  
  Refer to the [Maniskill Handbook](https://docs.google.com/document/d/1jZset2Qz7wtC8aKhkI5JZkMibM6piEr7Z8BliVoAPtc/edit?tab=t.3lcwjw24n671) in our Google Docs for guidelines and API references.
  
- **Runner Script:**  
  `/src/runner.py`

---

## Files to Submit

Please push your files under the `/src/{your_name}/` directory. Your submission should include:

- **Environment Code:**  
  `{your_env}.py` – your implementation of the custom environment.

- **Optional Explanation:**  
  `[optional] Explanations.md` – include this file if you have any incomplete aspects, mostly complete tasks, or additional notes you’d like us to consider.

---

## Submission Venue

Send your final submission (your name, branch name, and final commit ID) via email (as provided below) or WeChat DM.
- [lyitao17@gmail.com](mailto:lyitao17@gmail.com)
- [junhao1chen@gmail.com](mailto:junhao1chen@gmail.com)

---

Good luck, and we look forward to seeing your creative and innovative task designs!
