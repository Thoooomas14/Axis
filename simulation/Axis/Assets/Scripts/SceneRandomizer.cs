using UnityEngine;

public class SceneRandomizer : MonoBehaviour
{
    [Header("Object Randomizers")]
    public RandomStartPosition[] randomizers;

    public void randomizeScene() { foreach (var r in randomizers) r.randomize(); }
}
