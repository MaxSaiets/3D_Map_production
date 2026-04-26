/**
 * @jest-environment jsdom
 */
import { act, renderHook } from "@testing-library/react";
import { useGenerationStore } from "@/store/generation-store";

describe("GenerationStore", () => {
  beforeEach(() => {
    useGenerationStore.getState().reset();
  });

  it("initializes with the current default values", () => {
    const { result } = renderHook(() => useGenerationStore());

    expect(result.current.selectedArea).toBeNull();
    expect(result.current.isGenerating).toBe(false);
    expect(result.current.taskGroupId).toBeNull();
    expect(result.current.taskIds).toEqual([]);
    expect(result.current.activeTaskId).toBeNull();
    expect(result.current.progress).toBe(0);
    expect(result.current.roadWidthMultiplier).toBe(0.8);
    expect(result.current.exportFormat).toBe("3mf");
    expect(result.current.previewIncludeRoads).toBe(true);
  });

  it("stores selected area", () => {
    const bounds = { getNorth: () => 50.455 } as any;
    const { result } = renderHook(() => useGenerationStore());

    act(() => {
      result.current.setSelectedArea(bounds);
    });

    expect(result.current.selectedArea).toEqual(bounds);
  });

  it("updates generation flags", () => {
    const { result } = renderHook(() => useGenerationStore());

    act(() => {
      result.current.setGenerating(true);
      result.current.updateProgress(50, "Обробка...");
    });

    expect(result.current.isGenerating).toBe(true);
    expect(result.current.progress).toBe(50);
    expect(result.current.status).toBe("Обробка...");
  });

  it("updates task group metadata", () => {
    const { result } = renderHook(() => useGenerationStore());

    act(() => {
      result.current.setTaskGroup("group-1", ["task-a", "task-b"]);
    });

    expect(result.current.taskGroupId).toBe("group-1");
    expect(result.current.taskIds).toEqual(["task-a", "task-b"]);
    expect(result.current.activeTaskId).toBe("task-a");
  });

  it("updates geometry and preview settings", () => {
    const { result } = renderHook(() => useGenerationStore());

    act(() => {
      result.current.setRoadWidthMultiplier(1.5);
      result.current.setBuildingMinHeight(3);
      result.current.setTerrainEnabled(false);
      result.current.setPreviewIncludeWater(false);
    });

    expect(result.current.roadWidthMultiplier).toBe(1.5);
    expect(result.current.buildingMinHeight).toBe(3);
    expect(result.current.terrainEnabled).toBe(false);
    expect(result.current.previewIncludeWater).toBe(false);
  });

  it("resets to the initial state", () => {
    const { result } = renderHook(() => useGenerationStore());

    act(() => {
      result.current.setGenerating(true);
      result.current.setTaskGroup("group-1", ["task-a"]);
      result.current.updateProgress(50, "Test");
    });

    act(() => {
      result.current.reset();
    });

    expect(result.current.isGenerating).toBe(false);
    expect(result.current.taskGroupId).toBeNull();
    expect(result.current.taskIds).toEqual([]);
    expect(result.current.progress).toBe(0);
  });
});
