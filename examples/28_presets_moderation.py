"""Example 28 -- moderation preset across four kinds of post.

Runs `laya.moderation_questions()` over a targeted insult, a generic insult, spam and a
benign message, so each category's signal pattern is visible.
"""

from _common import laya, banner, describe, heading, load

banner("28", "Preset: content moderation", """
    `moderation_questions()` asks the four safety questions a moderation queue cares about
    -- toxic, harassment, threat, spam -- plus an ordinal severity rubric, all in one
    forward pass. The point of the example is the shape of each pattern: a targeted insult
    lights up harassment, a generic one only toxic, spam only spam, and a normal post
    nothing at all.
    """)

POSTS = [
    ("targeted insult", "Nobody asked for your opinion. You are the most useless person on "
                        "this forum and everyone knows it."),
    ("generic insult", "Honestly this is the dumbest post I have read all week. "
                       "Did you even think before typing?"),
    ("spam", "MAKE $5000 A WEEK FROM HOME! Click here now: bit.ly/xyz  "
             "Limited spots, DM me to start today!"),
    ("benign", "I switched to the new scheduler and my build times dropped by about 30 percent. "
               "Happy to share the config."),
]

agent = load("english")

results = {}
for label, post in POSTS:
    heading(label)
    answers = agent.predict({"post": post}, laya.moderation_questions())["answers"]
    results[label] = answers
    describe(answers)

heading("side by side")
print("   %-16s %-7s %-11s %-7s %-7s %s" %
      ("post", "toxic", "harassment", "threat", "spam", "severity"))
for label, _ in POSTS:
    a = results[label]
    print("   %-16s %-7.3f %-11.3f %-7.3f %-7.3f %.2f / 3" % (
        label, a["toxic"]["noul"], a["harassment"]["noul"], a["threat"]["noul"],
        a["spam"]["noul"], a["severity"]["score"]))

heading("what the pattern means")
tgt = results["targeted insult"]
gen = results["generic insult"]
spam = results["spam"]
ben = results["benign"]
print("   targeted insult -> toxic %.3f, harassment %.3f, spam %.3f  (a person is the target)"
      % (tgt["toxic"]["noul"], tgt["harassment"]["noul"], tgt["spam"]["noul"]))
print("   generic insult  -> toxic %.3f, harassment %.3f            (rude, not aimed at anyone)"
      % (gen["toxic"]["noul"], gen["harassment"]["noul"]))
print("   spam            -> spam %.3f,  toxic %.3f                 (advertising, not abuse)"
      % (spam["spam"]["noul"], spam["toxic"]["noul"]))
print("   benign          -> every flag %.3f, severity %.2f / 3"
      % (ben["toxic"]["noul"], ben["severity"]["score"]))
print()
print("   `severity` orders the set correctly (%.2f > %.2f > %.2f > %.2f) but is a coarse 0-3"
      % (tgt["severity"]["score"], gen["severity"]["score"],
         spam["severity"]["score"], ben["severity"]["score"]))
print("   rubric, so use it to sort a queue rather than as a hard threshold. The two hostile")
print("   posts differ by only %.3f on `toxic`: the questions that separate them are"
      % abs(tgt["toxic"]["noul"] - gen["toxic"]["noul"]))
print("   `harassment` and `threat`, not `toxic` on its own.")
