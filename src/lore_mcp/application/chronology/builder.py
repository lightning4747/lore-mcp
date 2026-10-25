"""Chronology builder implementation for narrative context (Tasks 9.2, 9.3, 9.4)."""

from collections import defaultdict
from collections.abc import Sequence

from lore_mcp.domain.chronology import (
    ChronologicalGroup,
    Evidence,
    OrderingStrategy,
    TemporalContext,
)
from lore_mcp.domain.types import ChunkId
from lore_mcp.ports.chronology import ChronologyBuilder


class DefaultChronologyBuilder(ChronologyBuilder):
    """Organizes retrieved evidence chronologically with uncertainty and
    conflict detection.
    """

    def build_context(
        self,
        evidence_list: Sequence[Evidence],
    ) -> TemporalContext:
        """Construct structured TemporalContext from retrieved evidence."""
        if not evidence_list:
            return TemporalContext(
                ordering="unknown",
                groups=[],
                timeline_notes=[],
                uncertain_ids=[],
                conflicts=[],
                has_uncertainty=False,
            )

        # 1. Group evidence by entity or event
        grouped_evidence: dict[str, list[Evidence]] = defaultdict(list[Evidence])
        for ev in evidence_list:
            key = str(ev.entity_id) if ev.entity_id else "general"
            grouped_evidence[key].append(ev)

        groups: list[ChronologicalGroup] = []
        all_uncertain_ids: list[ChunkId] = []
        all_conflicts: list[str] = []

        # Sort group keys for deterministic permutation invariance
        sorted_keys = sorted(grouped_evidence.keys())

        for key in sorted_keys:
            items = grouped_evidence[key]
            group_conflicts: list[str] = []
            group_uncertain_ids: list[ChunkId] = []

            # Detect conflicting narrative orders within the group
            loc_order_map: dict[tuple[int | None, int | None, int | None], int] = {}
            for item in items:
                if item.narrative_order is not None and (
                    item.season is not None or item.episode is not None
                ):
                    coord = (item.season, item.episode, item.scene_index)
                    existing_order = loc_order_map.get(coord)
                    if (
                        existing_order is not None
                        and existing_order != item.narrative_order
                    ):
                        desc = (
                            f"Conflicting narrative orders for {key} at "
                            f"S{item.season or '?'}E{item.episode or '?'}: "
                            f"{existing_order} vs {item.narrative_order}"
                        )
                        if desc not in group_conflicts:
                            group_conflicts.append(desc)
                        if item.chunk_id not in group_uncertain_ids:
                            group_uncertain_ids.append(item.chunk_id)
                    else:
                        loc_order_map[coord] = item.narrative_order

            # Classify ordering scheme for this group
            items_with_narrative = [
                it for it in items if it.narrative_order is not None
            ]
            items_with_broadcast = [
                it
                for it in items
                if it.season is not None and it.episode is not None
            ]

            ordered_items: list[Evidence] = []
            unordered_remainder: list[Evidence] = []
            group_ordering: OrderingStrategy = "unknown"

            if len(items_with_narrative) == len(items):
                # All items have narrative order
                group_ordering = "narrative"
                ordered_items = sorted(
                    items,
                    key=lambda it: (
                        it.narrative_order if it.narrative_order is not None else 0,
                        it.season if it.season is not None else 0,
                        it.episode if it.episode is not None else 0,
                        it.scene_index if it.scene_index is not None else 0,
                        str(it.chunk_id),
                    ),
                )
                unordered_remainder = []
            elif items_with_narrative:
                # Some items have narrative order, others do not
                group_ordering = "narrative"
                ordered_items = sorted(
                    items_with_narrative,
                    key=lambda it: (
                        it.narrative_order if it.narrative_order is not None else 0,
                        it.season if it.season is not None else 0,
                        it.episode if it.episode is not None else 0,
                        it.scene_index if it.scene_index is not None else 0,
                        str(it.chunk_id),
                    ),
                )
                missing_narrative = [
                    it for it in items if it.narrative_order is None
                ]
                unordered_remainder = sorted(
                    missing_narrative,
                    key=lambda it: (
                        it.season if it.season is not None else 0,
                        it.episode if it.episode is not None else 0,
                        it.scene_index if it.scene_index is not None else 0,
                        str(it.chunk_id),
                    ),
                )
                for it in unordered_remainder:
                    if it.chunk_id not in group_uncertain_ids:
                        group_uncertain_ids.append(it.chunk_id)
            elif len(items_with_broadcast) == len(items):
                # No narrative order, but all items have broadcast order
                group_ordering = "as_shown"
                ordered_items = sorted(
                    items,
                    key=lambda it: (
                        it.season if it.season is not None else 0,
                        it.episode if it.episode is not None else 0,
                        it.scene_index if it.scene_index is not None else 0,
                        str(it.chunk_id),
                    ),
                )
                unordered_remainder = []
            elif items_with_broadcast:
                # Some items have broadcast order, others do not
                group_ordering = "as_shown"
                ordered_items = sorted(
                    items_with_broadcast,
                    key=lambda it: (
                        it.season if it.season is not None else 0,
                        it.episode if it.episode is not None else 0,
                        it.scene_index if it.scene_index is not None else 0,
                        str(it.chunk_id),
                    ),
                )
                missing_broadcast = [
                    it
                    for it in items
                    if it.season is None or it.episode is None
                ]
                unordered_remainder = sorted(
                    missing_broadcast,
                    key=lambda it: str(it.chunk_id),
                )
                for it in unordered_remainder:
                    if it.chunk_id not in group_uncertain_ids:
                        group_uncertain_ids.append(it.chunk_id)
            else:
                # No temporal metadata on any items
                group_ordering = "unknown"
                ordered_items = []
                unordered_remainder = sorted(
                    items,
                    key=lambda it: str(it.chunk_id),
                )
                for it in unordered_remainder:
                    if it.chunk_id not in group_uncertain_ids:
                        group_uncertain_ids.append(it.chunk_id)

            group = ChronologicalGroup(
                key=key,
                ordered_items=ordered_items,
                unordered_remainder=unordered_remainder,
                ordering=group_ordering,
                uncertain_ids=group_uncertain_ids,
                conflicts=group_conflicts,
            )
            groups.append(group)

            for cid in group_uncertain_ids:
                if cid not in all_uncertain_ids:
                    all_uncertain_ids.append(cid)
            for conf in group_conflicts:
                if conf not in all_conflicts:
                    all_conflicts.append(conf)

        # 2. Determine overall ordering scheme
        group_orderings = {g.ordering for g in groups}
        if group_orderings == {"narrative"}:
            overall_ordering: OrderingStrategy = "narrative"
        elif "narrative" in group_orderings:
            overall_ordering = "narrative"
        elif "as_shown" in group_orderings:
            overall_ordering = "as_shown"
        else:
            overall_ordering = "unknown"

        # 3. Assemble sequential timeline notes
        timeline_notes: list[str] = []
        for group in groups:
            for it in group.ordered_items:
                loc: list[str] = []
                if it.season is not None:
                    loc.append(f"S{it.season}")
                if it.episode is not None:
                    loc.append(f"E{it.episode}")
                if it.scene_index is not None:
                    loc.append(f"scene {it.scene_index}")
                if it.narrative_order is not None:
                    loc.append(f"order {it.narrative_order}")
                loc_str = " ".join(loc) if loc else "timeline"

                snippet = it.content[:80] + ("..." if len(it.content) > 80 else "")
                timeline_notes.append(f"[{loc_str}] {snippet}")

            for it in group.unordered_remainder:
                snippet = it.content[:60] + ("..." if len(it.content) > 60 else "")
                timeline_notes.append(f"[unordered: {it.chunk_id}] {snippet}")

            for conf in group.conflicts:
                timeline_notes.append(f"[conflict] {conf}")

        return TemporalContext(
            ordering=overall_ordering,
            groups=groups,
            timeline_notes=timeline_notes,
            uncertain_ids=all_uncertain_ids,
            conflicts=all_conflicts,
            has_uncertainty=bool(all_uncertain_ids or all_conflicts),
        )
