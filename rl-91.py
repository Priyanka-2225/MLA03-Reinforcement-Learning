skill = 0.5

for lesson in range(20):
    difficulty = skill
    skill += 0.1*(1 - abs(skill - difficulty))
    print("Updated Skill:", skill)
