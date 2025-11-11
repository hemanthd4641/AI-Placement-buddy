import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).resolve().parents[1].parent
pkg = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
sys.path.append(str(pkg))

from Placement_Bot.modules.career_roadmap import CareerRoadmapGenerator

if __name__ == "__main__":
    cr = CareerRoadmapGenerator()
    roadmap = cr.generate_roadmap(
        target_role="Backend Developer",
        experience_level="beginner",
        timeline=12,
        primary_goal="Build a strong portfolio",
        current_skills=["Python", "SQL"]
    )
    print("PHASE COUNT:", len(roadmap.get('phases', [])))
    for i, ph in enumerate(roadmap.get('phases', []), 1):
        print(f"Phase {i} name:", ph.get('name'))
        print("  Projects:")
        for pj in ph.get('projects', []):
            print("   -", pj.get('name'), "| diff:", pj.get('difficulty'), "| tech:", pj.get('technologies'))
    print("META:", roadmap.get('metadata', {}).get('source'))
