import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.136.0/build/three.module.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let faceLandmarker;
let runningMode = "VIDEO";
const ROTATION_THRESHOLD = 5;
let isAligned = false;
let yaw = 0, pitch = 0, roll = 0;
let landmarks = []; 

// **Cargar el modelo de MediaPipe**
async function loadFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate: "GPU"
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });

    console.log("✅ FaceMesh cargado correctamente con Transformada");
    detectFace();
}

// **Detectar el rostro en cada frame**
async function detectFace() {
    if (!faceLandmarker) return;

    if (faceLandmarker.runningMode !== "VIDEO") {
        await faceLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    const results = await faceLandmarker.detectForVideo(video, performance.now());

    if (results.faceLandmarks.length > 0) {
        landmarks = results.faceLandmarks[0];

        if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
            const matrixData = results.facialTransformationMatrixes[0].data;
            isAligned = checkAlignment(matrixData);
        }
    } else {
        isAligned = false;
        landmarks = [];
    }

    drawOverlay(); // Dibujar video y puntos
    requestAnimationFrame(detectFace);
}

// **Verificar si la cabeza está alineada usando la matriz de transformación**
function checkAlignment(matrixData) {
    const matrix = new THREE.Matrix4().fromArray(matrixData);
    const quaternion = new THREE.Quaternion().setFromRotationMatrix(matrix);

    const euler = new THREE.Euler().setFromQuaternion(quaternion, 'YXZ');
    yaw = THREE.MathUtils.radToDeg(euler.y);
    pitch = THREE.MathUtils.radToDeg(euler.x);
    roll = THREE.MathUtils.radToDeg(euler.z);

    console.log(`Yaw: ${yaw.toFixed(2)}, Pitch: ${pitch.toFixed(2)}, Roll: ${roll.toFixed(2)}`);

    return Math.abs(yaw) < ROTATION_THRESHOLD && Math.abs(pitch) < ROTATION_THRESHOLD;
}

// **Dibujar overlay con video, información y puntos en el rostro**
function drawOverlay() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Limpiar canvas

    drawVideoFrame(); // Dibujar video en el canvas
    drawFacePoints(); // Dibujar puntos
    drawAlignmentIndicator(); // Mostrar círculo de alineación
    drawAngleText(); // Mostrar ángulos
}

// **Dibujar el video en el canvas**
function drawVideoFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

// **Dibujar puntos en la parte superior, inferior y costados del rostro**
function drawFacePoints() {
    if (landmarks.length === 0) return;

    const videoWidth = canvas.width;
    const videoHeight = canvas.height;

    const topPoint = landmarks[10];   // Frente
    const bottomPoint = landmarks[152]; // Barbilla
    const leftPoint = landmarks[234];  // Lado izquierdo
    const rightPoint = landmarks[454]; // Lado derecho

    const points = [
        { x: topPoint.x * videoWidth, y: topPoint.y * videoHeight },
        { x: bottomPoint.x * videoWidth, y: bottomPoint.y * videoHeight },
        { x: leftPoint.x * videoWidth, y: leftPoint.y * videoHeight },
        { x: rightPoint.x * videoWidth, y: rightPoint.y * videoHeight }
    ];

    ctx.fillStyle = "yellow"; // Color amarillo para los puntos
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// **Dibujar indicador de alineación en la esquina superior derecha**
function drawAlignmentIndicator() {
    const indicatorSize = 30;
    const margin = 10;
    const indicatorX = canvas.width - indicatorSize - margin;
    const indicatorY = margin + indicatorSize;

    ctx.beginPath();
    ctx.arc(indicatorX, indicatorY, indicatorSize / 2, 0, 2 * Math.PI);
    ctx.fillStyle = isAligned ? "limegreen" : "red";
    ctx.fill();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 2;
    ctx.stroke();
}

// **Dibujar los valores de Yaw, Pitch y Roll en pantalla**
function drawAngleText() {
    ctx.fillStyle = "white";
    ctx.font = "16px Arial";
    ctx.fillText(`Yaw: ${yaw.toFixed(2)}°`, 10, 20);
    ctx.fillText(`Pitch: ${pitch.toFixed(2)}°`, 10, 40);
    ctx.fillText(`Roll: ${roll.toFixed(2)}°`, 10, 60);
}

// **Conectar la cámara**
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    video.srcObject = stream;
    video.play();
});

// **Esperar a que el modelo cargue y comenzar la detección**
loadFaceLandmarker();
