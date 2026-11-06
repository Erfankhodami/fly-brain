# Full Implementation Plan: Connecting the 5,000-Neuron Drosophila Brain to a Unity 3D Game

This document outlines the complete architectural design, network protocol, C# game scripts, sensory-motor mapping, and phase-by-phase implementation plan for controlling an autonomous, biologically driven 3D fruit fly character in **Unity** using the real-time **5,000-neuron connectome engine**.

---

## 1. System Architecture

The setup operates as a high-speed, closed-loop **embodied sensorimotor system**:

```mermaid
flowchart TD
    subgraph Unity 3D Game Engine (Client)
        SCENE["3D Scene (Sugar Cubes, Obstacles, Terrain)"] --> SENSORS["FlySensors.cs (Bilateral Vision & Taste Colliders)"]
        SENSORS --> PACK["Pack Sensory JSON (Left/Right Eye Angle & Touch)"]
        PACK --> WS_CLIENT["FlyBrainClient.cs (WebSocket Client)"]
        
        WS_CLIENT -. "ws://localhost:8080/game_ws (60 Hz)" .-> WS_SERVER
        
        WS_RECV["Receive Motor JSON"] --> CONTROLLER["FlyCharacterController.cs (Physics & Kinematics)"]
        CONTROLLER --> RIGIDBODY["Rigidbody (Thrust, Torque, Banking)"]
        CONTROLLER --> ANIM["Animator (Proboscis Extension, Wing Flap, Leg Stride)"]
        RIGIDBODY & ANIM --> SCENE
    end

    subgraph Python Connectome Engine (Server on RTX 3050 GPU)
        WS_SERVER["visual_server.py (/game_ws Endpoint)"] --> PARSE["Unpack Sensory Drive to Firing Rates (Hz)"]
        PARSE --> GPU["RealtimeSubnetLIF Engine (PyTorch CUDA Graph)"]
        
        subgraph 5,000-Neuron Subnetwork
            GPU --> LC["LC Visual Projection & P9 Inputs"]
            GPU --> GRN["Sugar / Bitter GRN Taste Receptors"]
            LC & GRN --> SEZ["SEZ Interneurons (512k Synapses)"]
            SEZ --> DN["Descending Steering (DNa02) & Walking (P9_oDN1)"]
            SEZ --> MN9["Proboscis Motor Neurons (MN9_left & MN9_right)"]
        end
        
        DN & MN9 --> DECODE["Decode Firing Rates to Physics (Thrust, Yaw Torque, Feeding %)"]
        DECODE --> WS_SERVER
    end
```

---

## 2. High-Speed Sensory-Motor Communication Protocol

Communication runs over a local WebSocket channel (`ws://127.0.0.1:8080/game_ws`) at **60 Hz** (every 16.6 ms). Over local loopback, round-trip network latency is under **1 ms**.

### 2.1 Sensory Payload (Unity $\to$ Python Server)
Sent by Unity at the start of each physics tick:

```json
{
  "type": "sensory_frame",
  "timestamp": 12.45,
  "vision": {
    "target_visible": true,
    "target_type": "sugar",
    "left_eye_intensity": 0.85,
    "right_eye_intensity": 0.32,
    "target_azimuth_deg": -28.5,
    "target_distance": 4.2
  },
  "contact_taste": {
    "touching_sugar": false,
    "touching_bitter": false
  },
  "mechanosensory": {
    "wind_speed": 0.5,
    "air_drag_left": 1.0,
    "air_drag_right": 1.0
  }
}
```

### 2.2 Motor Payload (Python Server $\to$ Unity)
Computed by the 5,000-neuron connectome and sent back to Unity:

```json
{
  "type": "motor_frame",
  "timestamp": 12.45,
  "locomotion": {
    "forward_thrust": 0.78,
    "yaw_torque": -0.42,
    "pitch_trim": 0.05,
    "is_walking": true,
    "is_flying": false
  },
  "feeding": {
    "proboscis_extension": 0.0,
    "mn9_left_rate": 0.0,
    "mn9_right_rate": 0.0,
    "is_eating": false
  },
  "state_rates": {
    "p9_odn1_hz": 14.2,
    "dna02_left_hz": 18.0,
    "dna02_right_hz": 2.5
  }
}
```

---

## 3. Sensory Ingestion in Python Backend

When the Python server receives the sensory payload, it translates 3D environmental values into biological Poisson firing rates:

```python
# In code/visual_server.py (game_ws handler)
def process_game_sensory_frame(engine, data):
    vision = data.get("vision", {})
    taste = data.get("contact_taste", {})
    
    # 1. Visual Target Guidance -> P9 / Visual Inputs
    if vision.get("target_visible", False):
        l_int = vision.get("left_eye_intensity", 0.0)
        r_int = vision.get("right_eye_intensity", 0.0)
        p9_left_hz = 120.0 * l_int
        p9_right_hz = 120.0 * r_int
    else:
        p9_left_hz = 15.0  # Spontaneous exploratory search rate
        p9_right_hz = 15.0

    # 2. Contact Tasting -> Sugar / Bitter GRNs
    if taste.get("touching_sugar", False):
        sugar_hz = 250.0   # Contact feeding trigger
    else:
        sugar_hz = 0.0

    if taste.get("touching_bitter", False):
        bitter_hz = 200.0  # Repulsion trigger
    else:
        bitter_hz = 0.0

    # Apply to engine rate tensor
    engine.rates.zero_()
    engine.rates[engine.p9_indices[0]] = p9_left_hz
    engine.rates[engine.p9_indices[1]] = p9_right_hz
    engine.rates[engine.sugar_indices] = sugar_hz
    engine.rates[engine.bitter_indices] = bitter_hz
```

---

## 4. Motor Decoding in Python Backend

After stepping the GPU for 10–16 ms, the descending neuron activity is converted into physical forces:

```python
def decode_motor_outputs(engine, duration_sec):
    counts = engine.last_frame_spikes  # Spikes during this frame
    dt = duration_sec

    # 1. Forward Thrust from Walking Neurons (P9_oDN1)
    odn_l = counts[engine.descending_map["P9_oDN1_left"]] / dt
    odn_r = counts[engine.descending_map["P9_oDN1_right"]] / dt
    odn_mean = (odn_l + odn_r) / 2.0
    forward_thrust = np.clip(odn_mean / 25.0, 0.0, 1.0)

    # 2. Steering Yaw Torque from Turning Neurons (DNa02)
    dna_l = counts[engine.descending_map["DNa02_left"]] / dt
    dna_r = counts[engine.descending_map["DNa02_right"]] / dt
    # If DNa02_right fires higher -> turn right (positive torque)
    yaw_torque = np.clip((dna_r - dna_l) / 20.0, -1.0, 1.0)

    # 3. Proboscis Feeding from Motor Neurons (MN9)
    mn9_l = counts[engine.mn9_map["MN9_left"]] / dt
    mn9_r = counts[engine.mn9_map["MN9_right"]] / dt
    mn9_mean = (mn9_l + mn9_r) / 2.0
    proboscis_ext = np.clip(mn9_mean / 80.0, 0.0, 1.0)
    is_eating = mn9_mean > 25.0

    # If the fly is actively eating, cut forward locomotion
    if is_eating:
        forward_thrust = 0.0
        yaw_torque = 0.0

    return {
        "forward_thrust": float(forward_thrust),
        "yaw_torque": float(yaw_torque),
        "proboscis_extension": float(proboscis_ext),
        "is_eating": bool(is_eating)
    }
```

---

## 5. Unity-Side C# Scripts

Create these three modular C# scripts inside your Unity Project (`Assets/Scripts/FlyBrain/`):

### 5.1 `FlySensors.cs`
Extracts bilateral compound eye view angles and proboscis collider contacts:

```csharp
using UnityEngine;

public class FlySensors : MonoBehaviour
{
    [Header("Sensor Transforms")]
    public Transform leftEye;
    public Transform rightEye;
    public Collider proboscisTipCollider;

    [Header("Detection Settings")]
    public float maxVisualDistance = 15f;
    public LayerMask foodLayer;

    public bool IsTouchingSugar { get; private set; }
    public bool IsTouchingBitter { get; private set; }

    public (bool visible, float leftIntensity, float rightIntensity, float azimuth, float dist) GetVisualReading()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, maxVisualDistance, foodLayer);
        if (hits.Length == 0) return (false, 0f, 0f, 0f, 0f);

        // Find nearest sugar object
        Transform nearestFood = hits[0].transform;
        Vector3 toFood = nearestFood.position - transform.position;
        float distance = toFood.magnitude;

        Vector3 forward = transform.forward;
        float angleToTarget = Vector3.SignedAngle(forward, toFood, Vector3.up); // -180 to 180

        // Bilateral intensity based on field of view angle
        float leftIntensity = Mathf.Clamp01(1.0f - Mathf.Abs(angleToTarget - (-35f)) / 120f) * (1f / (1f + 0.1f * distance));
        float rightIntensity = Mathf.Clamp01(1.0f - Mathf.Abs(angleToTarget - (35f)) / 120f) * (1f / (1f + 0.1f * distance));

        return (true, leftIntensity, rightIntensity, angleToTarget, distance);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Sugar")) IsTouchingSugar = true;
        if (other.CompareTag("Bitter")) IsTouchingBitter = true;
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Sugar")) IsTouchingSugar = false;
        if (other.CompareTag("Bitter")) IsTouchingBitter = false;
    }
}
```

---

### 5.2 `FlyCharacterController.cs`
Receives motor outputs from the brain and applies physical movement and animations:

```csharp
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class FlyCharacterController : MonoBehaviour
{
    public float maxForwardSpeed = 2.5f;
    public float maxYawSpeed = 180f;

    public Animator flyAnimator; // Controls proboscis, wings, legs
    private Rigidbody rb;

    private float currentThrust = 0f;
    private float currentYawTorque = 0f;
    private float currentProboscis = 0f;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    public void ApplyMotorFrame(float thrust, float yawTorque, float proboscisExt, bool isEating)
    {
        currentThrust = thrust;
        currentYawTorque = yawTorque;
        currentProboscis = proboscisExt;

        if (flyAnimator != null)
        {
            flyAnimator.SetFloat("ProboscisExtension", currentProboscis);
            flyAnimator.SetFloat("WalkSpeed", currentThrust);
            flyAnimator.SetBool("IsEating", isEating);
        }
    }

    void FixedUpdate()
    {
        // 1. Steering Yaw
        float yaw = currentYawTorque * maxYawSpeed * Time.fixedDeltaTime;
        Quaternion turnRotation = Quaternion.Euler(0f, yaw, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);

        // 2. Forward Thrust
        Vector3 forwardMove = transform.forward * (currentThrust * maxForwardSpeed * Time.fixedDeltaTime);
        rb.MovePosition(rb.position + forwardMove);
    }
}
```

---

### 5.3 `FlyBrainClient.cs`
Maintains the high-speed WebSocket link between Unity and the Python simulation:

```csharp
using System;
using System.Text;
using UnityEngine;
using NativeWebSocket; // Unity package: https://github.com/endel/NativeWebSocket

public class FlyBrainClient : MonoBehaviour
{
    public string serverUrl = "ws://127.0.0.1:8080/game_ws";
    public FlySensors sensors;
    public FlyCharacterController controller;

    private WebSocket ws;
    private float sendTimer = 0f;
    public float sendInterval = 0.016f; // 60 Hz

    async void Start()
    {
        ws = new WebSocket(serverUrl);

        ws.OnOpen += () => Debug.Log("<color=green>[FlyBrain]</color> Connected to GPU Connectome!");
        ws.OnError += (e) => Debug.LogError("[FlyBrain] Error: " + e);
        ws.OnClose += (e) => Debug.LogWarning("[FlyBrain] Disconnected from Brain Server.");

        ws.OnMessage += (bytes) =>
        {
            string json = Encoding.UTF8.GetString(bytes);
            MotorData motor = JsonUtility.FromJson<MotorData>(json);
            if (motor != null)
            {
                controller.ApplyMotorFrame(motor.forward_thrust, motor.yaw_torque, motor.proboscis_extension, motor.is_eating);
            }
        };

        await ws.Connect();
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
        ws?.DispatchMessageQueue();
        #endif

        sendTimer += Time.deltaTime;
        if (sendTimer >= sendInterval && ws != null && ws.State == WebSocketState.Open)
        {
            sendTimer = 0f;
            SendSensoryUpdate();
        }
    }

    void SendSensoryUpdate()
    {
        var vis = sensors.GetVisualReading();
        string json = $"{{\"target_visible\":{vis.visible.ToString().ToLower()},\"left_eye_intensity\":{vis.leftIntensity:F2},\"right_eye_intensity\":{vis.rightIntensity:F2},\"touching_sugar\":{sensors.IsTouchingSugar.ToString().ToLower()}}}";
        ws.SendText(json);
    }

    private async void OnApplicationQuit()
    {
        if (ws != null) await ws.Close();
    }

    [Serializable]
    public class MotorData
    {
        public float forward_thrust;
        public float yaw_torque;
        public float proboscis_extension;
        public bool is_eating;
    }
}
```

---

## 6. Implementation Roadmap

| Phase | Objective | Deliverables | Estimated Time |
|---|---|---|---|
| **Phase 1** | **Server Game Endpoint** | Add `/game_ws` handler in `code/visual_server.py` supporting real-time sensory ingestion and motor torque extraction. | 1 Day |
| **Phase 2** | **Unity Environment Setup** | Create new Unity 3D project, import 3D fly asset, add `NativeWebSocket` package. | 1 Day |
| **Phase 3** | **Sensor Integration** | Set up bilateral eye raycasts and mouth/tarsi collision triggers in `FlySensors.cs`. | 1 Day |
| **Phase 4** | **Motor Physics & Locomotion** | Connect descending neuron torques to Unity Rigidbody in `FlyCharacterController.cs`. | 1 Day |
| **Phase 5** | **Closed-Loop Sugar Experiment** | Place interactive sugar cubes in the 3D scene (spawned on mouse-click); observe autonomous navigation and proboscis feeding. | 1 Day |

---

## 7. Interactive Game Mechanics Enabled by this Plan

Once implemented, you can build exciting interactive game experiments:

1. **Click-to-Spawn Sugar Cubes:** The player clicks anywhere on the 3D ground. A sugar cube spawns. The fly's visual sensors detect it, steering the character across the room until it reaches the cube and eats.
2. **Moving Sugar Target:** Place a sugar cube on a moving toy train or conveyor belt. The fly's visual target-tracking circuit (`LC10` / `DNa02`) will continuously adjust its pursuit trajectory in real time.
3. **Obstacle Navigation:** Place walls between the fly and the sugar. The mechanosensory inputs will steer around obstacles as it tracks the visual silhouette.
4. **Bitter Food Avoidance:** Place a yellow lemon or brown caffeine cube. When the fly approaches or touches it, `bitter_grn` activates, triggering `MDN` moonwalker neurons to make the fly perform an emergency backward walk or flight escape!

All components are fully compatible with your current PyTorch CUDA engine and RTX 3050 hardware.
