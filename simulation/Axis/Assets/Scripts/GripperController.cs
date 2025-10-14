using UnityEngine;

public class GripperController : MonoBehaviour
{
    [Header("Gripper Articulation Bodies")]
    public ArticulationBody leftKnuckleLink;
    public ArticulationBody rightKnuckleLink;
    public ArticulationBody leftInnerKnuckleLink;
    public ArticulationBody rightInnerKnuckleLink;
    public ArticulationBody leftFingerTip;
    public ArticulationBody rightFingerTip;

    [Header("Gripper Angle Multiplier")]
    public float KnuckleToInner = 1f;
    public float KnuckleToTip = -1f;

    [Header("Target Value")]
    public float target = 0f;

    // Update is called once per frame
    void Update()
    {
        var leftDrive = leftKnuckleLink.xDrive;
        leftDrive.target = Mathf.Min(target,leftDrive.upperLimit);
        leftKnuckleLink.xDrive = leftDrive;

        var leftInnerDrive = leftInnerKnuckleLink.xDrive;
        leftInnerDrive.target = leftKnuckleLink.xDrive.target * KnuckleToInner;
        leftInnerKnuckleLink.xDrive = leftInnerDrive;

        var leftTipDrive = leftFingerTip.xDrive;
        leftTipDrive.target = leftKnuckleLink.xDrive.target * KnuckleToTip;
        leftFingerTip.xDrive = leftTipDrive;

        var rightDrive = rightKnuckleLink.xDrive;
        rightDrive.target = Mathf.Max(-target, rightDrive.lowerLimit);
        rightKnuckleLink.xDrive = rightDrive;

        var rightInnerDrive = rightInnerKnuckleLink.xDrive;
        rightInnerDrive.target = rightKnuckleLink.xDrive.target * KnuckleToInner;
        rightInnerKnuckleLink.xDrive = rightInnerDrive;

        var rightTipDrive = rightFingerTip.xDrive;
        rightTipDrive.target = rightKnuckleLink.xDrive.target * KnuckleToTip;
        rightFingerTip.xDrive = rightTipDrive;
                
    }
}
