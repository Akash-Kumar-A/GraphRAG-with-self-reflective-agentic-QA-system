import json
t = open("dashboard_template.html", encoding="utf-8").read()
r = open("prepared/results.json", encoding="utf-8").read()
open("dps_dashboard.html", "w", encoding="utf-8").write(t.replace("/*DATA*/null", r))
print("wrote dps_dashboard.html")