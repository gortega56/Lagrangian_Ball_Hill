/*
Title: BallHill_Lagrangian
File Name: main.cpp
Copyright © 2015
Original authors: Gabriel Ortega
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description:

This program simulates a ball moving down a hill using Lagrangian Dynamics. Since we are not modeling friction
we only incorporate gravity which is a conservative force. The surface of the hill is elliptical giving us a position 
function:

	X = (x1, a2 - (x1/a1)^2 - (x3/a3)^2, x3)

Using the position function we can solve for velocity along the z dimension:

	x2' = - (2 * x1 * x1') / (a1^2) - (2 * x3 * x3') / (a3^2)

Now we can plug velocity into the Kinetic Energy Formula: T = 1/2 * mass * velocity^2

	T = m/2 * (x1'^2 + ((2 * x1 * x1') / (a1^2) + (2 * x3 * x3') / (a3^2))^2 + x3'^2

The relevant terms:

	dT/dx1			= (4*m*x'1/a1^2) * ((x1 * x1') / (a1^2) + (x3 * x3') / (a3^2))

	dT/dx3			= (4*m*x3'/a3^2) * ((x1 * x1') / (a1^2) + (x3 * x3') / (a3^2))

	dT/dx1'			= m * (x1' + 4*x1/a1^2 * ((x1 * x1') / (a1^2) + (x3 * x3') / (a3^2)))

	dT/dx3'			= m * (x3' + 4*x3/a3^2 * ((x1 * x1') / (a1^2) + (x3 * x3') / (a3^2)))

	d/dt(dT/dx1')	= m * (x1" + 4*x1/a1^2 * ((x1*x1"*x1'^2)/a1^2 + (x3*x3"*x3'^2)/a3^2) + 4*x1'/a1^2 * ((x1 * x1') / (a1^2) - (x3 * x3') / (a3^2)) 

	d/dt(dT/dx3')	= m * (x3" + 4*x3/a3^2 * ((x1*x1"*x1'^2)/a1^2 + (x3*x3"*x3'^2)/a3^2) + 4*x3'/a3^2 * ((x1 * x1') / (a1^2) - (x3 * x3') / (a3^2))

	dX/dx1			= (1, -2*x1/a1^2, 0)

	dX/dx3			= (0, -2*x3/a3^2, 1)

Remember the goal is to find the Lagrangian of the system: d/dt(dT/dDq) - dT/dq = Fq

	Fx1 = F * dX/dx1 = (-m*g*J) * dX/dx1 = 2*m*g*x1/a1^2		// * is a dot product when dealing with vector quantities

	Fx2 = F * dX/dx2 = (-m*g*J) * dX/dx2 = 2*m*g*x2/a2^2

We have both sides of the equation for x1 and x3:

	x1" + 4*x1/a1^2 * ((x1*x1"*x1'^2)/a1^2 + (x3*x3"*x3'^2)/a3^2)) = 2*m*g*x1/a1^2

	x3" + 4*x3/a3^2 * ((x1*x1"*x1'^2)/a1^2 + (x3*x3"*x3'^2)/a3^2)) = 2*m*g*x3/a3^2

This is a coupled system of second order linear equations which we must solve for DDx1 and DDx3. 
Using a trick to consolidate unknowns we define:
	
	y1 = x1/a1 --> x1 = y1*a1

	y3 = x3/a3 --> x3 = y3*a3

Using new variables we redefine the equations of motion in terms of y1 and y3:

	a1^2*y1"+4*y1*(y1*y1"+(y1')^2+y2*y2"+(y2')^2) = 2g*y1
	a2^2*y2"+4*y2*(y1*y1"+(y1')^2+y2*y2"+(y2')^2) = 2g*y2
	
Now we can solve the system in matrix form using the following terms:

  +   +   +                       +^{-1} +                             +
  |y1"| = |a1^2+4*y1^2 4*y1*y2    |      |2*g*y1-4*y1*((y1')^2+(y2')^2)|
  |y2"|   |4*y1*y2     a2^2+4*y2^2|      |2*g*y2-4*y2*((y1')^2+(y2')^2)|
  +   +   +                       +      +                             +

Resources:

	Game Physics by David Eberly
*/

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\common.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <chrono>
#include <vector>

#define PI															3.141592653589793238463f
#define PI_2														3.141592653589793238463f * 2.0f
#define GRAVITY														-0.00980665f
#define SPHERE_STACK_COUNT											10
#define SPHERE_SLICE_COUNT											10
#define PLANE_AX													25
#define PLANE_AY													10
#define PLANE_AZ													25
#define SPHERE_RADIUS												0.5f
#define WINDOW_WIDTH												1600
#define WINDOW_HEIGHT												1200

using namespace glm;

typedef std::chrono::high_resolution_clock::time_point				ClockTime;
typedef std::chrono::duration<double, std::milli>					Milliseconds;
typedef std::vector<vec3>											VertexBuffer;
typedef std::vector<vec3>											NormalBuffer;

GLFWwindow*	gWindow;
GLFWmonitor* gMonitor;
const GLFWvidmode* gVideoMode;
bool gShouldExit = 0;

GLuint gSphereVertexBufferID	=	0;
GLuint gPlaneVertexBufferID		=	0;
GLuint gSphereNormalBufferID	=	0;
GLuint gPlaneNormalBufferID		=	0;
GLuint gSphereVertexArrayID		=	0;
GLuint gPlaneVertexArrayID		=	0;
GLuint gShaderID				=	0;
GLuint gTransformID				=	0;
GLuint gColorID					=	0;

mat4 gProjection;
mat4 gView;
mat4 gModel;
mat4 gTransform;

VertexBuffer gSphereVertexBuffer;
VertexBuffer gPlaneVertexBuffer;

NormalBuffer gSphereNormalBuffer;
NormalBuffer gPlaneNormalBuffer;

vec3 gEyePosition		=	vec3(0.0f, PLANE_AY + 10.0f, 50.f);
vec3 gEyeDirection		=	vec3(0.0f, -1.0f, -1.0);
vec3 gEyeUp				=	vec3(0.0f, 1.0f, 0.0f);

float a3;
float a2;
float a1;
float gY1;
float gY2;
float gY1Dot;
float gY2Dot;

void InitializeSimulation();
void InitializeOpenGL();
void InitializeGeometry();
void InitializeShaders();
void InitializeProjectViewMatrices();
void BeginScene();
void BindGeometryAndShaders();
void UpdateScene(double millisecondsElapsed);
void RenderScene(double millisecondsElapsed);
void HandleInput();

void BuildSphere(VertexBuffer& vertexBuffer, NormalBuffer& normalBuffer, int stackCount, int sliceCount);
void BuildPlane(VertexBuffer& vertexBuffer, NormalBuffer& normalBuffer, float width, int vertexPerWidth, float depth, int vertexPerDepth);

struct Ball
{
	vec3 position;
	vec3 scale;
	mat4 transform;
} gBall;

struct Surface
{
	vec3 position;
	vec3 axis;
	mat4 transform;
} gSurface;

// Main Loop
int main(int argc, int* argv[])
{
	// Program Structure
	InitializeSimulation();
	InitializeOpenGL();
	InitializeGeometry();
	InitializeShaders();
	InitializeProjectViewMatrices();
	BeginScene();
	
	return 0;
}

void InitializeSimulation()
{
	
	gSurface.position	= { 0.0f, 0.0f, 0.0f };
	gSurface.transform = mat4(1.0f);
	gBall.position		= { 0.0f, PLANE_AY + SPHERE_RADIUS, 0.0f };
	gBall.scale			= { 1.0f, 1.0f, 1.0f };

	a3 = PLANE_AZ * 0.5f;
	a2 = PLANE_AY;
	a1 = PLANE_AX * 0.5f;
	gY1 = 0.0f;
	gY2 = 0.0f;
	gY1Dot = 0.1f;
	gY2Dot = 0.1f;
}

void InitializeOpenGL()
{
	// Graphics API setup.
	int glfwSuccess = glfwInit();
	if (!glfwSuccess) {
		exit(1);
	}

	// Create Window
	gMonitor = glfwGetPrimaryMonitor();
	gVideoMode = glfwGetVideoMode(gMonitor);

	//GLFWwindow* window = glfwCreateWindow(videoMode->width, videoMode->height, "Sphere", NULL, NULL);
	gWindow = glfwCreateWindow(1600, 1200, "Ball Hill Lagrangian", NULL, NULL);

	if (!gWindow) {
		glfwTerminate();
	}

	glfwMakeContextCurrent(gWindow);
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
}

void InitializeGeometry()
{
	// Calculate Vertices for a circle.
	BuildSphere(gSphereVertexBuffer, gSphereNormalBuffer, SPHERE_STACK_COUNT, SPHERE_SLICE_COUNT);
	
	// Bind vertex data to OpenGL
	glGenBuffers(1, &gSphereVertexBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, gSphereVertexBufferID); // OpenGL.GL_Array_Buffer = buffer with ID(vertexBufferID)
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * gSphereVertexBuffer.size(), &gSphereVertexBuffer[0], GL_STATIC_DRAW);

	glGenBuffers(1, &gSphereNormalBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, gSphereNormalBufferID); // OpenGL.GL_Array_Buffer = buffer with ID(vertexBufferID)
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * gSphereNormalBuffer.size(), &gSphereNormalBuffer[0], GL_STATIC_DRAW);

	// Build Plane
	BuildPlane(gPlaneVertexBuffer, gPlaneNormalBuffer, PLANE_AX, 20, PLANE_AZ, 20);

	glGenBuffers(1, &gPlaneVertexBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, gPlaneVertexBufferID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * gPlaneVertexBuffer.size(), &gPlaneVertexBuffer[0], GL_STATIC_DRAW);

	glGenBuffers(1, &gPlaneNormalBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, gPlaneNormalBufferID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * gPlaneNormalBuffer.size(), &gPlaneNormalBuffer[0], GL_STATIC_DRAW);
}

void InitializeShaders()
{
	// Extremely simple vertex and fragment shaders
	//const char* vertex_shader =
	//	"#version 400\n"
	//	"uniform mat4 transform;"
	//	"in vec3 vp;"
	//	"void main () {"
	//	"  gl_Position = transform * vec4 (vp, 1.0);"
	//	"}";

	const char* vertex_shader =
		"#version 400\n"
		"uniform mat4 transform;"
		"in vec3 vp;"
		"in vec3 vn;"
		"out vec3 pn;"
		"void main () {"
		"  pn = mat3(transform) * vn;"
		"  gl_Position = transform * vec4 (vp, 1.0);"
		"}";

	const char* fragment_shader =
		"#version 400\n"
		"uniform vec4 color;"
		"in vec3 pn;"
		"out vec4 frag_colour;"
		"void main () {"
		"  vec3 lightDir = vec3(0.0f, -1.0f, 0.0f);"
		"  frag_colour = color * dot(-lightDir, normalize(pn));"
		"}";

	GLuint vShaderID = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vShaderID, 1, &vertex_shader, NULL);
	glCompileShader(vShaderID);

	GLuint fShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fShaderID, 1, &fragment_shader, NULL);
	glCompileShader(fShaderID);

	gShaderID = glCreateProgram();
	glAttachShader(gShaderID, vShaderID);
	glAttachShader(gShaderID, fShaderID);
	glLinkProgram(gShaderID);

	// Bind Vertex Attributes
	GLuint vpID = glGetAttribLocation(gShaderID, "vp");
	GLuint vnID = glGetAttribLocation(gShaderID, "vn");

	// Sphere 
	glGenVertexArrays(1, &gSphereVertexArrayID);
	glBindVertexArray(gSphereVertexArrayID);
	glBindBuffer(GL_ARRAY_BUFFER, gSphereVertexBufferID);

	glVertexAttribPointer(vpID, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vpID);

	glVertexAttribPointer(vnID, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vnID);

	// Plane
	glGenVertexArrays(1, &gPlaneVertexArrayID);
	glBindVertexArray(gPlaneVertexArrayID);
	glBindBuffer(GL_ARRAY_BUFFER, gPlaneVertexBufferID);

	glVertexAttribPointer(vpID, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vpID);

	glVertexAttribPointer(vnID, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vnID);

	// Uniforms
	gTransformID = glGetUniformLocation(gShaderID, "transform");
	gColorID = glGetUniformLocation(gShaderID, "color");
}

void InitializeProjectViewMatrices()
{
	// Camera is static so only calculate projection and view matrix once.
	gProjection = perspective(PI * 0.25f, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 100.0f);
	gView = lookAt(gEyePosition, normalize(gSurface.position - gEyePosition), gEyeUp);
}

void BeginScene()
{
	// Loop setup. 
	ClockTime currentTime = std::chrono::high_resolution_clock::now();
	while (!gShouldExit)
	{
		ClockTime systemTime = std::chrono::high_resolution_clock::now();
		double deltaTime = Milliseconds(systemTime - currentTime).count() + DBL_EPSILON;
		currentTime = systemTime;

		UpdateScene(deltaTime);
		RenderScene(deltaTime);
		glfwSwapBuffers(gWindow);
	}
}

void UpdateScene(double millisecondsElapsed)
{
	if (millisecondsElapsed > 16.67f) 
	{
		millisecondsElapsed = 16.67f;
	}

	float time = (float)millisecondsElapsed;
	float deltaTime = time / 5.0f;
	float accumulator = deltaTime;

	float p1 = gY1;
	float p2 = gY2;
	float c1 = gY1 + 0.01f;
	float c2 = gY2 + 0.05f;

	while (accumulator < time)
	{
		float mat00 = a1 * a1 + 4.0f * gY1 * gY1;
		float mat11 = a3 * a3 + 4.0f * gY2 * gY2;
		float mat01 = 4.0f * gY1 * gY2;
		float inverseDeterminant = 1.0f / (mat00 * mat11 - mat01 * mat01);
		float inner = gY1Dot * gY1Dot + gY2Dot * gY2Dot;
		float y1NonHomogenous = (2.0f * GRAVITY * gY1) - (4.0f * gY1 * inner);
		float y2NonHomogenous = (2.0f * GRAVITY * gY2) - (4.0f * gY2 * inner);
		float y1DotDot = (mat11 * y1NonHomogenous - mat01 * y2NonHomogenous) * inverseDeterminant;
		float y2DotDot = (mat00 * y2NonHomogenous - mat01 * y1NonHomogenous) * inverseDeterminant;

		// Verlet Integration: xi+1 = 2 * xi - xi-1 * acceleration * dt^2
		gY1 = 2 * c1 - p1 + y1DotDot * deltaTime * deltaTime;
		gY2 = 2 * c2 - p2 + y2DotDot * deltaTime * deltaTime;

		p1 = c1;
		p2 = c2;
		c1 = gY1;
		c2 = gY2;

		float x = gY1 / a1;
		float z = gY2 / a3;

		vec3 position	= { gY1, (a2 - x * x - z * z), gY2 };
		vec3 normal		= { 2.0f * position.x / a1, 1.0f, 2.0f * position.z / a3 };
		normal			= normalize(normal);

		gBall.position = position + normal * SPHERE_RADIUS;

		// Perform integration
		accumulator += deltaTime;
	}

	// Update OGL bindings
	gBall.transform = scale(translate(mat4(1.0f), gBall.position), gBall.scale);

	// Update Title
	char title[100];
	float fps = 1000.0f / time;
	sprintf_s(title, "Ball Hill Lagrangian FPS: %f", fps);
	glfwSetWindowTitle(gWindow, title);
}


void HandleInput()
{
	vec3 force = vec3(0.0f);
	if (glfwGetKey(gWindow, GLFW_KEY_UP))
	{
		force.z -= 1.0f;
	}
	
	if (glfwGetKey(gWindow, GLFW_KEY_DOWN))
	{
		force.z += 1.0f;
	}
	
	if (glfwGetKey(gWindow, GLFW_KEY_RIGHT))
	{
		force.x += 1.0f;
	}
	
	if (glfwGetKey(gWindow, GLFW_KEY_LEFT))
	{
		force.x -= 1.0f;
	}

	vec3 acceleration = force / 0.5f;
	gY1Dot = acceleration.x * 0.01f;
	gY2Dot = acceleration.z * 0.01f;
}

void RenderScene(double millisecondsElapsed)
{
	// Clear buffers. Set shader.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glUseProgram(gShaderID);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	mat4 projView = gProjection * gView;
	mat4 sPVM = projView * gBall.transform;

	// Bind vertex layout.
	glBindVertexArray(gSphereVertexArrayID);
	
	// Bind Sphere vertex buffer.
	glBindBuffer(GL_ARRAY_BUFFER, gSphereVertexBufferID);

	// Bind uniform attributes.
	glUniformMatrix4fv(gTransformID, 1, GL_FALSE, &sPVM[0][0]);
	glUniform4f(gColorID, 0.5f, 0.5f, 0.5f, 1.0f);

	// Draw Sphere.
	glDrawArrays(GL_TRIANGLES, 0, gSphereVertexBuffer.size());

	mat4 pPVM = gProjection * gView * gSurface.transform;
	glBindVertexArray(gPlaneVertexArrayID);

	// Bind Plane vertex buffer.
	glBindBuffer(GL_ARRAY_BUFFER, gPlaneVertexBufferID);

	// Bind uniform attributes.
	glUniformMatrix4fv(gTransformID, 1, GL_FALSE, &pPVM[0][0]);
	glUniform4f(gColorID, 0.0f, 0.5f, 0.0f, 1.0f);

	// Draw Plane.
	glDrawArrays(GL_TRIANGLES, 0, gPlaneVertexBuffer.size());
}

void BuildSphere(VertexBuffer& vertexBuffer, NormalBuffer& normalBuffer, int stackCount, int sliceCount)
{
	std::vector<float> phiCoordinates;
	
	float phiStep = PI / stackCount;
	for (int i = 0; i < stackCount + 1; i++)
	{
		phiCoordinates.push_back((-PI_2) + (i * phiStep));
	}

	std::vector<float> thetaCoordinates;
	float thetaStep = (PI_2) / (sliceCount * 2);
	for (int i = 0; i < (sliceCount * 2) + 1; i++)
	{
		thetaCoordinates.push_back(0 + (i * thetaStep));
	}

	float radius = 0.5f;
	for (int i = 0; i < phiCoordinates.size() - 1; i++)
	{
		for (int j = 0; j < thetaCoordinates.size() - 1; j++)
		{
			vec3 vertex1 = vec3(radius * cosf(phiCoordinates[i]) * sinf(thetaCoordinates[j]), radius * sinf(phiCoordinates[i]) * sinf(thetaCoordinates[j]), radius * cosf(thetaCoordinates[j]));
			vec3 vertex2 = vec3(radius * cosf(phiCoordinates[i]) * sinf(thetaCoordinates[j + 1]), radius * sinf(phiCoordinates[i]) * sinf(thetaCoordinates[j + 1]), radius * cosf(thetaCoordinates[j + 1]));
			vec3 vertex3 = vec3(radius * cosf(phiCoordinates[i + 1]) * sinf(thetaCoordinates[j + 1]), radius * sinf(phiCoordinates[i + 1]) * sinf(thetaCoordinates[j + 1]), radius * cosf(thetaCoordinates[j + 1]));
			vec3 vertex4 = vec3(radius * cosf(phiCoordinates[i + 1]) * sinf(thetaCoordinates[j]), radius * sinf(phiCoordinates[i + 1]) * sinf(thetaCoordinates[j]), radius * cosf(thetaCoordinates[j]));

			vec3 vn = normalize(cross(vertex2 - vertex1, vertex3 - vertex1));
			if (thetaCoordinates[j] <= PI)
			{
				vertexBuffer.push_back(vertex1);
				vertexBuffer.push_back(vertex2);
				vertexBuffer.push_back(vertex3);

				vertexBuffer.push_back(vertex1);
				vertexBuffer.push_back(vertex3);
				vertexBuffer.push_back(vertex4);
			
				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);

				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);
			}

			if (thetaCoordinates[j] >= PI)
			{
				vertexBuffer.push_back(vertex1);
				vertexBuffer.push_back(vertex3);
				vertexBuffer.push_back(vertex2);

				vertexBuffer.push_back(vertex1);
				vertexBuffer.push_back(vertex4);
				vertexBuffer.push_back(vertex3);

				vn = -vn;
				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);

				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);
				normalBuffer.push_back(vn);
			}
		}
	}
}

void BuildPlane(VertexBuffer& vertexBuffer, NormalBuffer& normalBuffer, float width, int vertexPerWidth, float depth, int vertexPerDepth)
{
	// Test for minimum number of vertex
	if (vertexPerWidth < 2) vertexPerWidth = 2;
	if (vertexPerDepth < 2) vertexPerDepth = 2;

	// The distance between vertexes in a given axis:
	float widthStep = width / vertexPerWidth;
	float depthStep = depth / vertexPerDepth;

	float planeWidthDesloc = width / 2;
	float planeDepthDesloc = depth / 2;

	vec3 v1, v2, v3, vn;

	// Loop though the columns (z-axis)
	for (float k = 0; k < vertexPerDepth - 1; k++)
	{
		// Loop though the lines (x-axis)
		for (float i = 0; i < vertexPerWidth - 1; i++) // May need to change to vertexperwidth.
		{	// Creates a quad. Using two triangles
			// Vertices Position = (vertex position) - (plane dislocation)

				//Top Triangle
				// #1
			float x, z, sx, sz;
			x = (i + 1)*widthStep - planeWidthDesloc;
			z = k*depthStep - planeDepthDesloc;
			v1 = vec3(x, PLANE_AY - powf((x / planeWidthDesloc), 2) - powf((z / planeDepthDesloc), 2), z);
			vertexBuffer.push_back(v1);
			
			// #2
			x = i*widthStep - planeWidthDesloc;
			z = (k + 1)*depthStep - planeDepthDesloc;
			v2 = vec3(x, PLANE_AY - powf((x / planeWidthDesloc), 2) - powf((z / planeDepthDesloc), 2), z);
			vertexBuffer.push_back(v2);

			// #3
			x = i*widthStep - planeWidthDesloc;
			z = k*depthStep - planeDepthDesloc;
			v3 = vec3(x, PLANE_AY - powf((x / planeWidthDesloc), 2) - powf((z / planeDepthDesloc), 2), z);
			vertexBuffer.push_back(v3);

			vn = normalize(cross((v2 - v1), (v3 - v1)));
			normalBuffer.push_back(vn);
			normalBuffer.push_back(vn);
			normalBuffer.push_back(vn);

			//Bottom Triangle
			// #1
			x = (i + 1)*widthStep - planeWidthDesloc;
			z = (k + 1)*depthStep - (planeDepthDesloc);
			v1 = vec3(x, PLANE_AY - powf((x / planeWidthDesloc), 2) - powf((z / planeDepthDesloc), 2), z);
			vertexBuffer.push_back(v1);
		
			// #2
			x = i*widthStep - planeWidthDesloc;
			z = (k + 1)*depthStep - planeDepthDesloc;
			v2 = vec3(x, PLANE_AY - powf((x / planeWidthDesloc), 2) - powf((z / planeDepthDesloc), 2), z);
			vertexBuffer.push_back(v2);

			// #3
			x = (i + 1)*widthStep - planeWidthDesloc;
			z = k*depthStep - planeDepthDesloc;
			v3 = vec3(x, PLANE_AY - powf((x / planeWidthDesloc), 2) - powf((z / planeDepthDesloc), 2), z);
			vertexBuffer.push_back(v3);

			vn = normalize(cross((v2 - v1), (v3 - v1)));
			normalBuffer.push_back(vn);
			normalBuffer.push_back(vn);
			normalBuffer.push_back(vn);
		}
	}
}