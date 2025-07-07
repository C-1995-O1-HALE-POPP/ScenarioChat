from utils.dataset import SCENE_CATEGORY, SCENE_DATA

subtopics = []
for key in SCENE_CATEGORY:
    for theme in SCENE_DATA[key]["themes"]:
        subtopics.extend(theme["subtopics"])

print(len(subtopics))