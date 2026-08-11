import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mldb.store import RunStore

store = RunStore.from_env()
db = store.get_db()
db.attach(names=["val_metrics"], tags=["classification_1"])
with db.connect() as con:
    df = con.sql(
        """
        select "accuracy/validation" as accuracy, "model.init_args.encoder" as model, "model.init_args.pretrained_weights" as pretraining
        from val_metrics
        """
    ).df()
g = sns.barplot(
    df,
    x="model",
    y="accuracy",
    hue="pretraining",
)
g.set_title("Validation Accuracy by Model and Pretraining")
plt.savefig("classification_results.png", bbox_inches="tight")