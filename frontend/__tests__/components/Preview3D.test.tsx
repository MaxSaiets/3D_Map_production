/**
 * @jest-environment jsdom
 */
import { render, screen } from "@testing-library/react";
import { useThree } from "@react-three/fiber";
import { Preview3D } from "@/components/Preview3D";

jest.mock("@react-three/fiber", () => ({
  Canvas: () => <div data-testid="canvas" />,
  useFrame: jest.fn(),
  useThree: jest.fn(),
}));

jest.mock("@react-three/drei", () => ({
  OrbitControls: () => null,
  PerspectiveCamera: jest.fn().mockImplementation(() => null),
}));

const mockUseThree = useThree as jest.Mock;
const mockFetch = jest.fn();

beforeEach(() => {
  (global as any).fetch = mockFetch;
  mockFetch.mockReset();
  mockFetch.mockRejectedValue(new Error("fetch disabled in tests"));

  mockUseThree.mockReturnValue({
    camera: {
      quaternion: {
        x: 0,
        y: 0,
        z: 0,
        w: 1,
        clone() {
          return { x: 0, y: 0, z: 0, w: 1 };
        },
      },
      position: {
        addScaledVector: jest.fn(),
      },
      updateProjectionMatrix: jest.fn(),
    },
    gl: {
      domElement: {
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        ownerDocument: {
          addEventListener: jest.fn(),
          removeEventListener: jest.fn(),
          body: { style: {} },
        },
      },
    },
  });
});

describe("Preview3D", () => {
  it("renders the 3D preview canvas", () => {
    render(<Preview3D />);

    expect(screen.getByTestId("canvas")).toBeInTheDocument();
    expect(screen.getByText(/Швидке керування сценою/i)).toBeInTheDocument();
  });

  it("keeps the preview container styling", () => {
    const { container } = render(<Preview3D />);

    expect(container.firstChild).toHaveClass("relative", "h-full", "w-full", "bg-slate-950");
  });
});
