using UnityEngine;

public class PickAndPlaceController : MonoBehaviour
{
    [Header("UR3e Joints (shoulder to wrist)")]
    public ArticulationBody[] armJoints = new ArticulationBody[6];

    public float moveSpeed = 5f;

    [Header("Gripper Fingers")]
    public GripperController gripperController;
    public ArticulationBody leftFinger;
    public HingeJoint leftFingerTip;
    public ArticulationBody rightFinger;
    public HingeJoint rightFingerTip;

    [Header("Gripper Settings")]
    public float gripSpeed = 20f;
    public float gripForceThreshold = 5f; // Newtons
    private bool gripping = false;

    [Header("Pick & Place Waypoints (each must have a JointWaypoint component)")]
    public JointWaypoint aboveCube;
    public JointWaypoint atCube;
    public JointWaypoint aboveTarget;
    public JointWaypoint atTarget;

    private int step = 0;

    void FixedUpdate()
    {
        switch (step)
        {
            case 0: MoveToPose(aboveCube);  if (AtPose(aboveCube))  step++; break;
            case 1: MoveToPose(atCube);     if (AtPose(atCube))     step++; break;
            case 2: CloseGripper();         if (gripping)           step++; break;
            case 3: MoveToPose(aboveCube);  if (AtPose(aboveCube))  step++; break;
            case 4: MoveToPose(aboveTarget);if (AtPose(aboveTarget))step++; break;
            case 5: MoveToPose(atTarget);   if (AtPose(atTarget))   step++; break;
            case 6: OpenGripper(); break;
        }
    }

    void MoveToPose(JointWaypoint waypoint)
    {
        if (waypoint == null || waypoint.jointAngles == null)
        {
            Debug.LogWarning("Waypoint is missing or has no joint angles!");
            return;
        }

        for (int i = 0; i < armJoints.Length; i++)
        {
            if (armJoints[i] == null){
                Debug.LogWarning($"Joint {i} not assigned!");
                continue;
            }

            var drive = armJoints[i].xDrive;
            
            float current = armJoints[i].jointPosition[0] * Mathf.Rad2Deg;
            float targetAngle = waypoint.jointAngles[i];
            float delta = Mathf.DeltaAngle(current, targetAngle);
            // if (delta < 0) delta = Mathf.Max(delta, moveSpeed * -1 * Time.deltaTime);
            // else if (delta >=0) delta = Mathf.Min(delta, moveSpeed * Time.deltaTime);

            drive.target += delta*0.01f;
            armJoints[i].xDrive = drive;

            Debug.Log($"Moving joint {i}: Delta: {delta:F1} | Current: {current:F1}° → Current Target Setting: {drive.target:F1}° (target {targetAngle:F1}°)");
        }
    }


    bool AtPose(JointWaypoint waypoint)
    {
        if (waypoint == null || waypoint.jointAngles == null) return false;

        for (int i = 0; i < armJoints.Length; i++)
        {
            float current = armJoints[i].jointPosition[0] * Mathf.Rad2Deg;
            float target = waypoint.jointAngles[i];
            if (Mathf.Abs(Mathf.DeltaAngle(current, target)) > 0.1f)
                return false;
        }
        return true;
    }

    void CloseGripper()
    {

        float leftForce = Mathf.Abs(leftFinger.driveForce[0]);
        float rightForce = Mathf.Abs(rightFinger.driveForce[0]);

        Debug.Log($"Moving Grippers: Force Left {leftForce:F5} | Force Right {rightForce:F5}");

        if (leftForce > gripForceThreshold || rightForce > gripForceThreshold)
        {
            gripping = true;
            return;
        }

        
        gripperController.target += gripSpeed * Time.deltaTime;
    }

    void OpenGripper()
    {
        gripperController.target = 0;
        // var leftDrive = leftFinger.xDrive;
        // leftDrive.target -= gripSpeed * Time.fixedDeltaTime;
        // leftFinger.xDrive = leftDrive;

        // var rightDrive = rightFinger.xDrive;
        // rightDrive.target += gripSpeed * Time.fixedDeltaTime;
        // rightFinger.xDrive = rightDrive;
    }

    void OnCollisionEnter(Collision collision)
    {
        if(step == 2)
        {
            gripping = true;
            Debug.Log($"Impulse {collision.impulse} | Relative Velocity {collision.relativeVelocity}");
        }
    }
}
