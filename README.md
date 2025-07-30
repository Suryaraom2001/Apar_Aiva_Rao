 Detects humans using a YOLOv8 model.
- Sorts images into:
  - `human/` if a person is detected.
  - `spray/` if the filename contains "spray".
  - `not_human/` if no person is detected.
- Uses OpenCV and Ultralytics.








