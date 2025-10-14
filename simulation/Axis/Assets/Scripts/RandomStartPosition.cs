using Unity.Mathematics;
using UnityEngine;

public class RandomStartPosition : MonoBehaviour
{
    [Header("Random Bounds")]
    [Header("Min")]
    public Vector3 min;
    [Header("Max")]
    public Vector3 max;

    [Header("Rotation")]
    public Vector3 rotationMin;
    public Vector3 rotationMax;

    public void randomize()
    {
        var xPos = UnityEngine.Random.Range(min.x, max.x);
        var yPos = UnityEngine.Random.Range(min.y, max.y);
        var zPos = UnityEngine.Random.Range(min.z, max.z);

        var xRot = UnityEngine.Random.Range(rotationMin.x, rotationMax.x);
        var yRot = UnityEngine.Random.Range(rotationMin.y, rotationMax.y);
        var zRot = UnityEngine.Random.Range(rotationMin.z, rotationMax.z);

        Vector3 position = new Vector3(xPos, yPos, zPos);
        Quaternion rotation = Quaternion.Euler(new Vector3(xRot,yRot,zRot));
        transform.SetPositionAndRotation(position, rotation);
    }    
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        randomize();
    }
}
