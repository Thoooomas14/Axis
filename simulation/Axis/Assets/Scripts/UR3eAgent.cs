using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// Handles joint and gripper motion using degrees.
/// Works directly with your GripperController script.
/// </summary>
public class UR3eAgent : Agent
{
    [Header("UR3e Arm Joints (shoulder to wrist3)")]
    public ArticulationBody[] armJoints = new ArticulationBody[6];

    [Header("Gripper Controller")]
    public GripperController gripper; // ← drag your gripper object here

    [Header("Scene Objects")]
    public Transform cube;
    public Transform plate;

    [Header("Movement Settings")]
    public float jointStepDeg = 2.0f; // degrees per action step
    public float gripperStepDeg = 2.0f; // degrees per open/close action

    [Header("Randomizer")]
    public SceneRandomizer randomizer; // optional script with public ResetEnv()

    private const int NumJoints = 6;
    private const int NumActions = 7; // 6 arm + 1 gripper axis

    public override void Initialize()
    {
        foreach (var j in armJoints)
        {
            if (j == null) continue;
            var drive = j.xDrive;
            drive.stiffness = Mathf.Max(drive.stiffness, 20000f);
            drive.damping = Mathf.Max(drive.damping, 300f);
            drive.forceLimit = Mathf.Max(drive.forceLimit, 1000f);
            j.xDrive = drive;
        }
    }

    public override void OnEpisodeBegin()
    {
        // Reset joints to zero configuration
        foreach (var j in armJoints)
        {
            if (j == null) continue;
            var drive = j.xDrive;
            drive.target = 0f;
            j.xDrive = drive;
            j.TeleportRoot(j.transform.position, j.transform.rotation);
        }

        // Reset environment
        if (randomizer != null)
        {
            randomizer.randomizeScene();
        }

        // Reset gripper
        if (gripper != null)
            gripper.target = 0f;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect joint states in degrees
        foreach (var j in armJoints)
        {
            if (j == null)
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                continue;
            }
            float posDeg = j.jointPosition[0] * Mathf.Rad2Deg;
            float velDeg = j.jointVelocity[0] * Mathf.Rad2Deg;
            sensor.AddObservation(posDeg);
            sensor.AddObservation(velDeg);
        }

        // Gripper opening
        sensor.AddObservation(gripper != null ? gripper.target : 0f);

        // Object positions relative to robot base
        if (cube != null && plate != null)
        {
            Vector3 cubeRel = transform.InverseTransformPoint(cube.position);
            Vector3 plateRel = transform.InverseTransformPoint(plate.position);
            sensor.AddObservation(cubeRel);
            sensor.AddObservation(plateRel);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(Vector3.zero);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        if (a.Length < NumActions) return;

        // --- Arm control ---
        for (int i = 0; i < NumJoints; i++)
        {
            if (armJoints[i] == null) continue;
            float delta = Mathf.Clamp(a[i], -1f, 1f) * jointStepDeg;
            var drive = armJoints[i].xDrive;
            drive.target = Mathf.Clamp(drive.target + delta, drive.lowerLimit, drive.upperLimit);
            armJoints[i].xDrive = drive;
        }

        // --- Gripper control ---
        if (gripper != null)
        {
            float gripCmd = Mathf.Clamp(a[NumJoints], -1f, 1f);
            gripper.target = Mathf.Clamp(gripper.target + gripCmd * gripperStepDeg, -20f, 20f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        a.Clear();

        // Simple manual control for testing
        a[0] = Input.GetKey(KeyCode.UpArrow) ? 1f : (Input.GetKey(KeyCode.DownArrow) ? -1f : 0f);
        a[NumJoints] = Input.GetKey(KeyCode.Space) ? 1f : (Input.GetKey(KeyCode.LeftControl) ? -1f : 0f);
    }
}
