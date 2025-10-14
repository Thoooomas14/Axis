using UnityEngine;

public class JointWaypoint : MonoBehaviour
{
    public float[] jointAngles = new float[6];
#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(transform.position, 0.01f);
    }
#endif
}
