import json
t = open("dashboard_template.html").read(); r = open("prepared/results.json").read()
open("dps_dashboard.html", "w").write(t.replace("/*DATA*/null", r))
