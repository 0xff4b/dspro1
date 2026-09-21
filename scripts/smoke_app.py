"""Exercise the UI and every shipped model without external address services."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(Path("src/app.py").resolve()), default_timeout=120).run()
assert not app.exception, [x.message for x in app.exception]
assert not app.error, [x.value for x in app.error]
models = list(app.sidebar.selectbox[0].options)
assert models and "best_model_v3" in models[0], models
for index, label in enumerate(models):
    app.sidebar.selectbox[0].select_index(index).run()
    assert not app.exception, [x.message for x in app.exception]
    assert not app.error, [x.value for x in app.error]
    assert app.sidebar.metric, f"No prediction for {label}"
    assert not app.sidebar.warning, [x.value for x in app.sidebar.warning]
    print(f"PASS {label}: {app.sidebar.metric[0].value}")
app.radio[0].set_value("Model Performance").run()
assert not app.exception, [x.message for x in app.exception]
assert not app.error, [x.value for x in app.error]
app.radio[0].set_value("Modell & Daten").run()
assert not app.exception, [x.message for x in app.exception]
assert not app.error, [x.value for x in app.error]
print("PASS all pages")
