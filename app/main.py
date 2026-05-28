"""
Aplicación GUI para ejecutar el pipeline de análisis SNIES.

Esta aplicación proporciona una interfaz gráfica simple para usuarios no técnicos.
En la primera ejecución, solicita al usuario seleccionar la carpeta raíz del proyecto.
"""

from __future__ import annotations

import json
import datetime
import os
import shutil
import sys
import threading
import time
import tkinter as tk
import pandas as pd
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Dependencia ya usada por el resto del proyecto (ETL)
# pandas se importa lazy donde se necesita (no al inicio para acelerar arranque)
from typing import Callable

# Añadir el directorio raíz al path si es necesario
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Imports ligeros al inicio (solo config básico)
from etl.config import (
    get_base_dir,
    set_base_dir,
    update_paths_for_base_dir,
    ARCHIVO_NORMALIZACION,
    MODELS_DIR,
    get_smlmv_sesion,
    set_smlmv_sesion,
    set_benchmark_costo,
)
# HISTORIC_DIR se importa dentro de run_pipeline después de set_base_dir para asegurar que esté inicializado

# Imports pesados se hacen lazy (solo cuando se ejecuta el pipeline)
# Esto acelera el arranque de la aplicación


def _open_in_excel(path: Path) -> None:
    """Abre un archivo con la app por defecto (Excel en Windows)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    # Windows: abre con la aplicación predeterminada
    os.startfile(str(path))  # type: ignore[attr-defined]


def _open_default_app(path: Path) -> None:
    """Abre un archivo con la app predeterminada del sistema."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    os.startfile(str(path))  # type: ignore[attr-defined]


def _open_text_file(path: Path) -> None:
    """Abre un archivo de texto (fallback a app por defecto si no hay notepad)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception:
        # Último fallback
        _open_default_app(path)


def _ask_yes_no(title: str, msg: str, parent: tk.Misc | None = None) -> bool:
    try:
        return bool(messagebox.askyesno(title, msg, parent=parent))
    except Exception:
        return False


# Usar etl.config como única fuente de verdad para rutas exe/script
def _get_config_file_for_gui() -> Path:
    """Ruta de config.json (delega en etl.config)."""
    from etl.config import get_config_file_path
    return get_config_file_path()


def get_pipeline_lock_file() -> Path:
    from etl.normalizacion import ARCHIVO_PROGRAMAS  # Lazy import
    return ARCHIVO_PROGRAMAS.parent / ".pipeline.lock"


def get_lock_age_seconds(lock_file: Path) -> float | None:
    if not lock_file.exists():
        return None
    try:
        return time.time() - lock_file.stat().st_mtime
    except Exception:
        return None


LOCK_STALE_SECONDS = 60 * 30  # 30 minutos


def explain_file_in_use() -> str:
    return (
        "No se pudo escribir el archivo porque está abierto o bloqueado.\n\n"
        "Cierra Excel / Power BI (y cualquier visor del archivo) y vuelve a intentarlo."
    )


def bind_tooltip(widget: tk.Misc, text: str, delay_ms: int = 450) -> None:
    """Muestra un tooltip flotante al pasar el mouse (sin dependencias extra)."""
    state: dict[str, tk.Toplevel | str | None] = {"win": None, "after_id": None}

    def _destroy() -> None:
        aid = state.get("after_id")
        if aid is not None:
            try:
                widget.after_cancel(aid)
            except (tk.TclError, ValueError, TypeError):
                pass
            state["after_id"] = None
        tw = state.get("win")
        if tw is not None:
            try:
                tw.destroy()
            except tk.TclError:
                pass
            state["win"] = None

    def _show() -> None:
        _destroy()
        try:
            x = int(widget.winfo_rootx()) + 12
            y = int(widget.winfo_rooty()) + int(widget.winfo_height()) + 6
        except tk.TclError:
            return
        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        state["win"] = tw
        tw.configure(bg="#1e293b")
        tk.Label(
            tw,
            text=text,
            justify="left",
            background="#1e293b",
            foreground="#f8fafc",
            font=("Segoe UI", 9),
            padx=10,
            pady=8,
            wraplength=300,
        ).pack()

    def _schedule(_event=None) -> None:
        _destroy()
        state["after_id"] = widget.after(delay_ms, _show)

    widget.bind("<Enter>", _schedule, add="+")
    widget.bind("<Leave>", lambda _e: _destroy(), add="+")
    widget.bind("<ButtonPress>", lambda _e: _destroy(), add="+")


def safe_messagebox_error(title: str, msg: str, parent: tk.Misc | None = None) -> None:
    try:
        messagebox.showerror(title, msg, parent=parent)
    except Exception:
        # fallback
        print(f"[ERROR] {title}: {msg}")


def can_write_file(path: Path) -> bool:
    """
    Retorna True si el archivo puede abrirse en modo escritura (append).
    Útil para detectar Excel/PowerBI bloqueando el archivo en Windows.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            return True
    except PermissionError:
        return False
    except Exception:
        # Si no existe o hay otro error, no bloqueamos por defecto.
        return True


def validar_entorno_pipeline() -> tuple[bool, list[str]]:
    """
    Comprueba que el entorno esté listo para ejecutar el pipeline (archivos, permisos).
    Returns:
        (ok, lista de mensajes: faltantes o errores si not ok; vacía o ["Todo listo"] si ok)
    """
    from etl.config import (
        ARCHIVO_NORMALIZACION,
        OUTPUTS_DIR,
        get_archivo_referentes,
        get_archivo_catalogo_eafit,
    )
    mensajes: list[str] = []
    if not ARCHIVO_NORMALIZACION.exists():
        mensajes.append(f"Falta archivo de normalización: {ARCHIVO_NORMALIZACION}")
    try:
        if not get_archivo_referentes().exists():
            mensajes.append("Falta archivo de referentes (ref/referentesUnificados.xlsx o .csv)")
    except FileNotFoundError:
        mensajes.append("Falta archivo de referentes (ref/referentesUnificados.xlsx o .csv)")
    try:
        if not get_archivo_catalogo_eafit().exists():
            mensajes.append("Falta archivo de catálogo EAFIT (ref/catalogoOfertasEAFIT.xlsx o .csv)")
    except FileNotFoundError:
        mensajes.append("Falta archivo de catálogo EAFIT (ref/catalogoOfertasEAFIT.xlsx o .csv)")
    test_file = OUTPUTS_DIR / ".write_test"
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        test_file.write_text("", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        mensajes.append(f"No se puede escribir en la carpeta outputs: {e}")
    ok = len(mensajes) == 0
    if ok:
        mensajes.append("Todo listo para ejecutar el pipeline.")
    return ok, mensajes


def validate_programas_schema(path_xlsx: Path) -> tuple[bool, str]:
    """
    Valida el "schema mínimo" requerido para que el pipeline funcione.
    """
    import pandas as pd  # Lazy import
    
    required_cols = [
        "CÓDIGO_SNIES_DEL_PROGRAMA",
        "NOMBRE_DEL_PROGRAMA",
        "NOMBRE_INSTITUCIÓN",
        "NIVEL_DE_FORMACIÓN",
    ]
    try:
        df_head = pd.read_excel(path_xlsx, sheet_name="Programas", nrows=5)
    except Exception as exc:
        return False, f"No se pudo leer la hoja 'Programas' en {path_xlsx.name}: {exc}"
    missing = [c for c in required_cols if c not in df_head.columns]
    if missing:
        return (
            False,
            "El archivo descargado no tiene las columnas mínimas esperadas.\n"
            f"Faltan: {', '.join(missing)}\n\n"
            "Esto puede indicar que SNIES cambió el formato o que la descarga no es válida.",
        )
    return True, "OK"


def get_configured_base_dir() -> Path | None:
    """
    Devuelve el directorio base del proyecto calculado automáticamente
    desde la ubicación del ejecutable. No lee config.json para evitar
    que rutas absolutas de otros usuarios rompan la detección.
    """
    try:
        base_dir = get_base_dir()
        if base_dir and base_dir.exists() and base_dir.is_dir():
            return base_dir
    except Exception:
        pass
    return None


def ensure_base_dir(parent_window: tk.Misc | None = None, prompt_if_missing: bool = True) -> Path | None:
    """
    Devuelve el directorio base del proyecto detectado automáticamente
    desde la ubicación del ejecutable. No muestra diálogos de selección
    de carpeta — el programa se configura solo para cualquier usuario
    que ejecute el .exe desde la carpeta correcta del proyecto en OneDrive.

    Si no se puede detectar la carpeta, muestra un error claro explicando
    que el exe no está en el lugar correcto.
    """
    base_dir = get_configured_base_dir()

    if not base_dir:
        if parent_window is not None:
            messagebox.showerror(
                "Error de configuración",
                "No se pudo detectar la carpeta del proyecto.\n\n"
                "Verifique que SniesManager.exe esté en la carpeta raíz "
                "del proyecto, junto a las carpetas:\n"
                "  - ref/\n  - outputs/\n  - models/\n  - docs/\n\n"
                "No mueva el ejecutable fuera de esa carpeta.\n"
                "Si el problema persiste, use el botón 'Cambiar carpeta' "
                "en la pantalla principal.",
                parent=parent_window,
            )
        return None

    try:
        update_paths_for_base_dir(base_dir)
    except Exception as exc:
        if parent_window is not None:
            messagebox.showerror(
                "Error",
                f"No se pudo inicializar las rutas del proyecto:\n\n{exc}",
                parent=parent_window,
            )
        return None

    return base_dir


# ========== COLORIMETRÍA EAFIT (paleta base oficial mejorada) ==========
# Blanco puro #FFFFFF | Negro puro #000000
# Azul Zafre oscuro #000066 (Pantone 2738C) | Azul Azure claro #00A9E0 (Pantone 306C)
# Combinaciones sugeridas: texto azul zafre sobre blanco, blanco sobre azul zafre, azul azure como acento.
EAFIT = {
    "white": "#FFFFFF",
    "black": "#000000",
    "azul_zafre": "#000066",   # Azul Zafre oscuro - principal, botones primarios, títulos
    "azul_azure": "#00A9E0",   # Azul Azure claro - acentos, enlaces, secundario
    "azul_azure_light": "#4DC8F0",  # Azul Azure más claro para hover
    "bg": "#F5F7FA",           # Fondo general mejorado (gris muy claro, más moderno)
    "card_bg": "#FFFFFF",
    "card_border": "#E1E8ED",  # Borde sutil para tarjetas
    "card_shadow": "#D1D9E0",  # Color para sombra simulada
    "text": "#1A1A1A",         # Negro suavizado para mejor legibilidad
    "text_on_dark": "#FFFFFF", # Texto sobre azul zafre
    "text_muted": "#6B7280",   # Gris mejorado para texto secundario
    "text_light": "#9CA3AF",   # Gris muy claro para hints
    "success": "#10B981",      # Verde mejorado
    "success_light": "#D1FAE5", # Verde claro para fondos
    "warning": "#F59E0B",      # Amarillo mejorado
    "warning_light": "#FEF3C7", # Amarillo claro para fondos
    "danger": "#EF4444",       # Rojo mejorado
    "danger_light": "#FEE2E2", # Rojo claro para fondos
    "info": "#3B82F6",         # Azul info
    "info_light": "#DBEAFE",   # Azul claro para fondos
    "separator": "#E5E7EB",    # Separadores sutiles
}
THEME = EAFIT  # alias para no romper referencias


def apply_modern_style(root: tk.Tk) -> None:
    """
    Aplica la colorimetría EAFIT mejorada y estilos ttk modernos en toda la aplicación.
    Paleta: Blanco #FFFFFF, Negro #000000, Azul Zafre #000066, Azul Azure #00A9E0.
    Incluye mejoras estéticas: mejor espaciado, bordes sutiles, tipografía mejorada.
    """
    style = ttk.Style(root)
    # "clam" permite personalizar colores de botones (vista en Windows no siempre)
    for theme in ("clam", "vista", "default"):
        try:
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        except Exception:
            continue

    base_bg = EAFIT["bg"]
    root.configure(bg=base_bg)

    # Frames mejorados con mejor diseño
    style.configure("App.TFrame", background=base_bg)
    style.configure(
        "Card.TFrame",
        background=EAFIT["card_bg"],
        relief="flat",
        borderwidth=1,
    )
    # Intentar agregar borde sutil a las tarjetas (si el tema lo permite)
    try:
        style.map("Card.TFrame", 
                  bordercolor=[("", EAFIT["card_border"])],
                  lightcolor=[("", EAFIT["card_border"])],
                  darkcolor=[("", EAFIT["card_border"])])
    except tk.TclError:
        pass
    style.configure("Page.TFrame", background=base_bg)
    
    # Frame con borde superior para separadores elegantes
    # Nota: height puede causar problemas en algunos temas, lo removemos
    try:
        style.configure("Separator.TFrame", background=EAFIT["separator"])
    except tk.TclError:
        # Fallback si el estilo no funciona
        pass

    # Labels mejorados con mejor tipografía y espaciado
    style.configure(
        "Header.TLabel",
        background=base_bg,
        font=("Segoe UI", 24, "bold"),
        foreground=EAFIT["azul_zafre"],
    )
    style.configure(
        "SubHeader.TLabel",
        background=base_bg,
        foreground=EAFIT["text_muted"],
        font=("Segoe UI", 11),
    )
    style.configure(
        "SectionTitle.TLabel",
        background=EAFIT["card_bg"],
        font=("Segoe UI", 13, "bold"),
        foreground=EAFIT["azul_zafre"],
    )
    style.configure(
        "Muted.TLabel",
        background=EAFIT["card_bg"],
        foreground=EAFIT["text_muted"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "Light.TLabel",
        background=EAFIT["card_bg"],
        foreground=EAFIT["text_light"],
        font=("Segoe UI", 8),
    )
    style.configure("Path.TLabel", background=base_bg, foreground=EAFIT["text"], font=("Segoe UI", 9))
    style.configure("Status.TLabel", background=base_bg, foreground=EAFIT["text_muted"], font=("Segoe UI", 9))
    style.configure("Help.TLabel", background=base_bg, foreground=EAFIT["text_muted"], font=("Segoe UI", 9))
    
    # Labels de estado con colores
    style.configure("Success.TLabel", background=base_bg, foreground=EAFIT["success"], font=("Segoe UI", 9, "bold"))
    style.configure("Warning.TLabel", background=base_bg, foreground=EAFIT["warning"], font=("Segoe UI", 9, "bold"))
    style.configure("Danger.TLabel", background=base_bg, foreground=EAFIT["danger"], font=("Segoe UI", 9, "bold"))
    style.configure("Info.TLabel", background=base_bg, foreground=EAFIT["info"], font=("Segoe UI", 9, "bold"))

    # Botones mejorados con mejor diseño y padding
    style.configure(
        "Primary.TButton",
        font=("Segoe UI", 11, "bold"),
        padding=(20, 12),
    )
    try:
        style.configure(
            "Primary.TButton",
            background=EAFIT["azul_zafre"],
            foreground=EAFIT["text_on_dark"],
        )
        style.map("Primary.TButton", 
                  background=[("active", EAFIT["azul_azure"]), 
                             ("pressed", EAFIT["azul_zafre"])])
    except tk.TclError:
        pass  # tema vista no permite background en TButton
    style.configure(
        "Secondary.TButton",
        font=("Segoe UI", 10),
        padding=(16, 10),
    )
    try:
        style.configure("Secondary.TButton", foreground=EAFIT["azul_zafre"])
        style.map("Secondary.TButton", 
                  foreground=[("active", EAFIT["azul_azure"])])
    except tk.TclError:
        pass
    # Botón de peligro (cancelar)
    style.configure("Danger.TButton", font=("Segoe UI", 10), padding=(16, 10))
    try:
        style.configure("Danger.TButton", foreground=EAFIT["white"], background=EAFIT["danger"])
    except tk.TclError:
        style.configure("Danger.TButton", foreground=EAFIT["danger"])
    try:
        style.map("Danger.TButton", foreground=[("active", EAFIT["danger_light"])])
    except tk.TclError:
        pass
    style.configure("Back.TButton", font=("Segoe UI", 10), padding=(12, 8))
    try:
        style.configure("Back.TButton", foreground=EAFIT["azul_azure"])
        style.map("Back.TButton", 
                  foreground=[("active", EAFIT["azul_zafre"])])
    except tk.TclError:
        pass
    
    # Botón pequeño para utilidades
    style.configure("Small.TButton", font=("Segoe UI", 9), padding=(10, 6))
    try:
        style.configure("Small.TButton", foreground=EAFIT["azul_zafre"])
    except tk.TclError:
        pass


class EditableTable(ttk.Frame):
    """
    Tabla editable simple basada en ttk.Treeview.
    Pensada para ediciones puntuales (no millones de filas).
    """

    def __init__(
        self,
        master: tk.Misc,
        columns: list[str],
        height: int = 15,
        editable_columns: set[str] | None = None,
        on_change: Callable[[int, str, str], None] | None = None,
        dropdown_values: dict[str, list[str]] | None = None,
    ):
        super().__init__(master)
        self.columns = columns
        self._data: list[dict] = []
        self._item_to_index: dict[str, int] = {}
        self.editable_columns = editable_columns if editable_columns is not None else set(columns)
        self.on_change = on_change
        # IMPORTANTE: Usar la MISMA referencia del dict para que las actualizaciones se reflejen
        # Si se pasa None, crear un nuevo dict vacío
        if dropdown_values is not None:
            self.dropdown_values = dropdown_values  # Usar la referencia pasada (compartida)
        else:
            self.dropdown_values = {}  # Crear nuevo dict vacío solo si no se pasó ninguno

        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=height)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=160, anchor="w", minwidth=100)

        # Configurar grid con scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configurar pesos para que se expandan correctamente
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # Fila del scrollbar horizontal
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)  # Columna del scrollbar vertical

        self._editor: tk.Entry | ttk.Combobox | None = None
        self.tree.bind("<Double-1>", self._begin_edit)
        self._min_height_rows = 5
        self._row_height_px = 24

    def set_height_from_pixels(self, pixels: int) -> None:
        """Ajusta la altura del Treeview en número de filas según el espacio en píxeles (responsive)."""
        if pixels < 50:
            return
        try:
            rows = max(self._min_height_rows, pixels // self._row_height_px)
            self.tree.configure(height=rows)
        except (tk.TclError, ValueError):
            pass

    def set_rows(self, rows: list[dict]) -> None:
        self._data = rows
        self._item_to_index.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, row in enumerate(rows):
            values = [row.get(c, "") for c in self.columns]
            item_id = self.tree.insert("", "end", values=values)
            self._item_to_index[item_id] = idx

    def get_rows(self) -> list[dict]:
        return self._data

    def get_selected_index(self) -> int | None:
        sel = self.tree.selection()
        if not sel:
            return None
        return self._item_to_index.get(sel[0])

    def delete_selected(self) -> None:
        idx = self.get_selected_index()
        if idx is None:
            return
        # borrar del modelo
        self._data.pop(idx)
        # reconstruir todo para mantener índices consistentes
        self.set_rows(self._data)
    
    def set_cell_value(self, row_idx: int, column: str, value: str) -> None:
        """Establece el valor de una celda específica y actualiza la visualización."""
        if row_idx < 0 or row_idx >= len(self._data):
            return
        if column not in self.columns:
            return
        
        # Actualizar datos
        self._data[row_idx][column] = value
        
        # Actualizar visualización
        item_ids = list(self.tree.get_children())
        if row_idx < len(item_ids):
            item_id = item_ids[row_idx]
            col_index = self.columns.index(column)
            current_vals = list(self.tree.item(item_id, "values"))
            if col_index < len(current_vals):
                current_vals[col_index] = value
                self.tree.item(item_id, values=current_vals)

    def add_row(self, default_row: dict | None = None) -> None:
        row = {c: "" for c in self.columns}
        if default_row:
            row.update(default_row)
        self._data.append(row)
        self.set_rows(self._data)

    def _begin_edit(self, event):
        # Identificar celda
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)  # e.g. "#1"
        if not row_id or not col_id:
            return
        col_index = int(col_id.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(self.columns):
            return
        column = self.columns[col_index]
        if column not in self.editable_columns:
            return
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return

        idx = self._item_to_index.get(row_id)
        if idx is None:
            return

        # destruir editor anterior
        if self._editor is not None:
            try:
                self._editor.destroy()
            except Exception:
                pass
            self._editor = None

        x, y, w, h = bbox
        value = str(self._data[idx].get(column, "") or "")

        # Para ES_REFERENTE, usar Combobox en lugar de Entry para mejor UX
        if column == "ES_REFERENTE":
            editor = ttk.Combobox(self.tree, values=["Sí", "No"], state="readonly", width=w//8)
            editor.set(value if value in ["Sí", "No"] else "No")
            editor.focus_set()
            editor.place(x=x, y=y, width=w, height=h)
            
            def commit(_evt=None):
                new_val = editor.get()
                self._data[idx][column] = new_val
                # actualizar visualmente
                current_vals = list(self.tree.item(row_id, "values"))
                current_vals[col_index] = new_val
                self.tree.item(row_id, values=current_vals)
                if self.on_change is not None:
                    try:
                        self.on_change(idx, column, new_val)
                    except Exception:
                        pass
                editor.destroy()
                self._editor = None
            
            editor.bind("<<ComboboxSelected>>", commit)
            editor.bind("<FocusOut>", commit)
            editor.bind("<Return>", commit)
            editor.bind("<Escape>", lambda e: (editor.destroy(), setattr(self, '_editor', None)))
        elif column in self.dropdown_values:
            # Para columnas con valores personalizados (dropdown), usar Combobox
            dropdown_options = self.dropdown_values[column]
            if not dropdown_options:
                # Si no hay opciones, usar Entry normal como fallback
                editor = tk.Entry(self.tree)
                editor.insert(0, value)
                editor.select_range(0, tk.END)
                editor.focus_set()
                editor.place(x=x, y=y, width=w, height=h)
                
                def commit(_evt=None):
                    new_val = editor.get()
                    self._data[idx][column] = new_val
                    current_vals = list(self.tree.item(row_id, "values"))
                    current_vals[col_index] = new_val
                    self.tree.item(row_id, values=current_vals)
                    if self.on_change is not None:
                        try:
                            self.on_change(idx, column, new_val)
                        except Exception:
                            pass
                    editor.destroy()
                    self._editor = None
                
                editor.bind("<Return>", commit)
                editor.bind("<FocusOut>", commit)
                editor.bind("<Escape>", lambda e: (editor.destroy(), setattr(self, '_editor', None)))
            else:
                # Crear Combobox con las opciones del catálogo
                editor = ttk.Combobox(self.tree, values=dropdown_options, state="readonly", width=50)
                # Buscar el valor actual en las opciones (puede estar vacío o tener un valor)
                current_value = value.strip() if value else ""
                if current_value in dropdown_options:
                    editor.set(current_value)
                else:
                    editor.set("")  # Si no está en las opciones, dejar vacío
                editor.focus_set()
                # Hacer el dropdown más ancho para mostrar nombres completos (mínimo 500px para ver mejor)
                # También ajustar posición si es necesario para que no se salga de la ventana
                dropdown_width = max(w, 500)
                editor.place(x=x, y=y, width=dropdown_width, height=h)
                # Abrir el dropdown automáticamente para mejor UX
                editor.event_generate('<Button-1>')
                editor.event_generate('<Down>')
                
                def commit(_evt=None):
                    new_val = editor.get()
                    self._data[idx][column] = new_val
                    # actualizar visualmente
                    current_vals = list(self.tree.item(row_id, "values"))
                    current_vals[col_index] = new_val
                    self.tree.item(row_id, values=current_vals)
                    if self.on_change is not None:
                        try:
                            self.on_change(idx, column, new_val)
                        except Exception:
                            pass
                    editor.destroy()
                    self._editor = None
                
                editor.bind("<<ComboboxSelected>>", commit)
                editor.bind("<FocusOut>", commit)
                editor.bind("<Return>", commit)
                editor.bind("<Escape>", lambda e: (editor.destroy(), setattr(self, '_editor', None)))
        else:
            # Para otras columnas, usar Entry normal
            editor = tk.Entry(self.tree)
            editor.insert(0, value)
            editor.select_range(0, tk.END)
            editor.focus_set()
            editor.place(x=x, y=y, width=w, height=h)

            def commit(_evt=None):
                new_val = editor.get()
                self._data[idx][column] = new_val
                # actualizar visualmente
                current_vals = list(self.tree.item(row_id, "values"))
                current_vals[col_index] = new_val
                self.tree.item(row_id, values=current_vals)
                if self.on_change is not None:
                    try:
                        self.on_change(idx, column, new_val)
                    except Exception:
                        pass
                editor.destroy()
                self._editor = None

            editor.bind("<Return>", commit)
            editor.bind("<FocusOut>", commit)
            editor.bind("<Escape>", lambda e: (editor.destroy(), setattr(self, '_editor', None)))
        
        self._editor = editor


class ManualReviewPage(ttk.Frame):
    """Edición manual de emparejamientos (falsos positivos) en Programas.xlsx."""

    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        
        # Import lazy de pandas y ARCHIVO_PROGRAMAS (solo cuando se abre esta página)
        import pandas as pd
        from etl.normalizacion import ARCHIVO_PROGRAMAS
        
        self.base_dir = ensure_base_dir(self)
        if not self.base_dir:
            if on_back:
                on_back()
            return

        self.file_path = ARCHIVO_PROGRAMAS
        # Columnas principales (vista resumida)
        self.main_columns = [
            "CÓDIGO_SNIES_DEL_PROGRAMA",
            "NOMBRE_INSTITUCIÓN",
            "NOMBRE_DEL_PROGRAMA",
            "NIVEL_DE_FORMACIÓN",
            "PROGRAMA_NUEVO",
            "ES_REFERENTE",
            "PROBABILIDAD",
            "PROGRAMA_EAFIT_CODIGO",
            "PROGRAMA_EAFIT_NOMBRE",
        ]
        # Columnas visibles actuales (por defecto todas, se actualiza al cargar)
        self.display_columns = self.main_columns.copy()
        # Todas las columnas disponibles (se establece al cargar el archivo)
        self.all_columns: list[str] = []
        # Estado de la vista: True = completa (todas), False = principal (9 columnas)
        self.view_complete = True
        self.editable_columns = {
            "ES_REFERENTE",
            "PROGRAMA_EAFIT_CODIGO",
            "PROGRAMA_EAFIT_NOMBRE",
            "PROBABILIDAD",
        }
        # df_view: solo columnas necesarias para mostrar/filtrar (no se usa para guardar en disco)
        # Type hint usa string para evitar importar pandas al inicio
        self.df_view = None  # type: ignore
        self._filtered_df: pd.DataFrame | None = None
        self.page_size = 200
        self.page_index = 0
        # Cambios pendientes persistentes entre páginas (por código SNIES normalizado)
        self.pending_updates: dict[str, dict[str, object]] = {}
        # Backup oculto antes de guardar (para restaurar si es necesario)
        self.last_backup_path: Path | None = None
        
        # Cargar catálogo EAFIT para dropdown de programas (inicializar variables, se carga después de crear msg)
        self.catalogo_eafit_df = None
        self.programas_eafit_nombres = []
        self.programas_eafit_dict = {}  # Mapeo nombre -> código

        header = ttk.Frame(self, padding=12, style="Page.TFrame")
        header.pack(fill=tk.X)
        header_left = ttk.Frame(header, style="Page.TFrame")
        header_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(header_left, text="✏️ Ajuste manual de emparejamientos", style="Header.TLabel").pack(anchor="w")
        self.subheader_label = ttk.Label(header_left, text="Edita ES_REFERENTE y programa EAFIT. Los cambios se guardan en Programas.xlsx.", style="SubHeader.TLabel")
        self.subheader_label.pack(anchor="w", pady=(4, 0), fill=tk.X)
        if on_back:
            ttk.Button(header, text="← Volver al menú", command=lambda: on_back() if on_back else None, style="Back.TButton").pack(side=tk.RIGHT)

        # Botones principales en grid para mejor organización responsive
        btns = ttk.Frame(self, padding="10")
        btns.pack(fill=tk.X)
        btns.grid_columnconfigure(0, weight=1, uniform="btn_col")
        btns.grid_columnconfigure(1, weight=1, uniform="btn_col")
        btns.grid_columnconfigure(2, weight=1, uniform="btn_col")
        
        # Fila 1: Acciones principales
        row1 = ttk.Frame(btns, style="App.TFrame")
        row1.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Button(row1, text="Recargar Programas.xlsx", command=self._load).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row1, text="Abrir en Excel", command=self._open_excel).pack(side=tk.LEFT, padx=6)
        self.btn_save = ttk.Button(row1, text="Guardar cambios (todo)", command=self._save)
        self.btn_save.pack(side=tk.LEFT, padx=6)
        self.btn_delete = ttk.Button(row1, text="Descartar cambios (fila)", command=self._discard_row_changes)
        self.btn_delete.pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Descartar todo", command=self._discard_all_changes).pack(side=tk.LEFT, padx=6)
        self.btn_toggle_view = ttk.Button(row1, text="Vista principal", command=self._toggle_view, state=tk.DISABLED)
        self.btn_toggle_view.pack(side=tk.LEFT, padx=6)
        
        # Fila 2: Marcado y restauración
        row2 = ttk.Frame(btns, style="App.TFrame")
        row2.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Button(row2, text="Marcar SÍ referente", command=self._mark_si_referente).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row2, text="Marcar NO referente", command=self._mark_no_referente).pack(side=tk.LEFT, padx=6)
        self.btn_restore = ttk.Button(row2, text="Restaurar estado anterior", command=self._restore_backup, state=tk.DISABLED)
        self.btn_restore.pack(side=tk.LEFT, padx=6)
        
        # Fila 3: Filtros y búsqueda
        row3 = ttk.Frame(btns, style="App.TFrame")
        row3.grid(row=2, column=0, columnspan=3, sticky="ew")
        ttk.Label(row3, text="Filtro:").pack(side=tk.LEFT, padx=(0, 6))
        self.filter_var = tk.StringVar(value="SOLO_NUEVOS")
        filter_combo = ttk.Combobox(
            row3,
            textvariable=self.filter_var,
            state="readonly",
            values=["SOLO_NUEVOS", "SOLO_REFERENTES", "TODOS"],
            width=18,
        )
        filter_combo.pack(side=tk.LEFT)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())
        ttk.Button(row3, text="Aplicar filtro", command=self._apply_filter).pack(side=tk.LEFT, padx=6)
        ttk.Label(row3, text="Buscar:").pack(side=tk.LEFT, padx=(14, 6))
        self.search_var = tk.StringVar(value="")
        self.search_entry = ttk.Entry(row3, textvariable=self.search_var, width=22)
        self.search_entry.pack(side=tk.LEFT)
        ttk.Button(row3, text="Buscar", command=self._apply_filter).pack(side=tk.LEFT, padx=6)
        ttk.Label(row3, text="Nivel:").pack(side=tk.LEFT, padx=(14, 6))
        self.nivel_filter_var = tk.StringVar(value="TODOS")
        self.nivel_filter_combo = ttk.Combobox(
            row3,
            textvariable=self.nivel_filter_var,
            state="readonly",
            values=["TODOS", "ESPECIALIZACIÓN", "MAESTRÍA", "PREGRADO", "TÉCNICO", "TECNOLÓGICO", "DOCTORADO"],
            width=20,
        )
        self.nivel_filter_combo.pack(side=tk.LEFT)
        self.nivel_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        pager = ttk.Frame(self, padding=(10, 0, 10, 10))
        pager.pack(fill=tk.X)
        self.page_label = ttk.Label(pager, text="Página: -", foreground=EAFIT["text_muted"])
        self.page_label.pack(side=tk.LEFT)
        self.selection_label = ttk.Label(pager, text="Selección: -", foreground=EAFIT["text_muted"])
        self.selection_label.pack(side=tk.LEFT, padx=(12, 0))
        self.pending_label = ttk.Label(pager, text="Cambios pendientes: 0", foreground=EAFIT["text_muted"])
        self.pending_label.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(pager, text="Anterior", command=self._prev_page).pack(side=tk.RIGHT)
        ttk.Button(pager, text="Siguiente", command=self._next_page).pack(side=tk.RIGHT, padx=6)

        # Banner de solo lectura
        self.readonly_banner = ttk.Label(
            self,
            text="",
            foreground=EAFIT["danger"],
            font=("Segoe UI", 9, "bold"),
        )
        self.readonly_banner.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.msg = tk.Text(self, height=6, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9), bg=EAFIT["card_bg"], fg=EAFIT["text"])
        self.msg.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Preparar valores iniciales para dropdown de PROGRAMA_EAFIT_NOMBRE (vacío por ahora)
        # IMPORTANTE: Crear el dict ANTES de pasarlo a EditableTable para poder actualizarlo después
        # Usar una referencia compartida para que las actualizaciones se reflejen
        self.dropdown_values_dict = {}
        
        self.table = EditableTable(
            self,
            columns=self.display_columns,
            height=18,
            editable_columns=self.editable_columns,
            on_change=self._on_cell_change,
            dropdown_values=self.dropdown_values_dict,  # Pasar la referencia al dict (compartido)
        )
        self.table.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        # Actualizar estado de selección
        self.table.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Ahora que msg está creado, cargar catálogo EAFIT y actualizar dropdown
        self._cargar_catalogo_eafit()
        # Actualizar dropdown_values después de cargar el catálogo
        # IMPORTANTE: Actualizar el mismo dict que se pasó a la tabla (referencia compartida)
        if self.programas_eafit_nombres:
            # Actualizar el dict compartido (esto actualizará automáticamente self.table.dropdown_values)
            self.dropdown_values_dict["PROGRAMA_EAFIT_NOMBRE"] = self.programas_eafit_nombres
            self._log(f"✓ Dropdown de PROGRAMA_EAFIT_NOMBRE configurado con {len(self.programas_eafit_nombres)} opciones")
            self._log(f"💡 Haz doble clic en la columna PROGRAMA_EAFIT_NOMBRE para ver el dropdown con todos los programas EAFIT")
        else:
            self._log("⚠️ No se pudieron cargar programas EAFIT para el dropdown")
        
        self._log("Tip: edita una celda con doble clic. Ajusta ES_REFERENTE y PROGRAMA_EAFIT_* si hay falsos positivos.")
        # Auto-cargar si existe el archivo (mejor UX).
        if self.file_path.exists():
            self._load()
        else:
            self._log("No se encontró outputs/Programas.xlsx. Ejecuta primero el análisis SNIES (Pipeline).")

        # Monitor de lock del pipeline
        self._poll_lock()
        # Atajos de teclado (bind al root)
        root = self.winfo_toplevel()
        root.bind("<Control-f>", lambda _e: self._focus_search())
        root.bind("<Control-s>", lambda _e: self._save())
        root.bind("<Delete>", lambda _e: self._discard_row_changes())
        self._log("Nota: solo puedes editar ES_REFERENTE / PROGRAMA_EAFIT_* / PROBABILIDAD. Los demás campos son de SNIES (solo lectura).")

    def _on_resize(self, w: int, h: int) -> None:
        """Responsive: ajusta la altura de la tabla y wraplengths al espacio disponible."""
        # Aproximado: header + botones + paginador + banner + msg ~ 320 px
        table_pixels = max(120, h - 320)
        self.table.set_height_from_pixels(table_pixels)
        
        # Ajustar wraplength del subheader dinámicamente
        if hasattr(self, 'subheader_label'):
            wraplength = max(400, w - 100)
            self.subheader_label.config(wraplength=wraplength)

    def _focus_search(self):
        try:
            self.search_entry.focus_set()
            self.search_entry.selection_range(0, tk.END)
        except Exception:
            pass

    def _norm_codigo(self, v: object) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    def _now_iso(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _touch_pending(self):
        try:
            self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
        except Exception:
            pass

    def _cargar_catalogo_eafit(self):
        """Carga el catálogo EAFIT para el dropdown de programas."""
        try:
            from etl.clasificacionProgramas import cargar_catalogo_eafit
            import pandas as pd
            self.catalogo_eafit_df = cargar_catalogo_eafit()
            
            # Obtener nombres de programas EAFIT (solo activos ya están filtrados)
            if 'Nombre Programa EAFIT' in self.catalogo_eafit_df.columns:
                # Intentar obtener código también
                posibles_columnas_codigo = ['Codigo EAFIT', 'Código Programa', 'CODIGO_PROGRAMA', 'Codigo Programa']
                columna_codigo = None
                for col in posibles_columnas_codigo:
                    if col in self.catalogo_eafit_df.columns:
                        columna_codigo = col
                        break
                
                # Crear lista de nombres y diccionario nombre -> código
                nombres = self.catalogo_eafit_df['Nombre Programa EAFIT'].astype(str).tolist()
                self.programas_eafit_nombres = sorted(set(nombres))  # Ordenar y eliminar duplicados
                
                if columna_codigo:
                    # Crear diccionario nombre -> código
                    for _, row in self.catalogo_eafit_df.iterrows():
                        nombre = str(row['Nombre Programa EAFIT'])
                        codigo = str(row[columna_codigo]) if pd.notna(row[columna_codigo]) else ""
                        if nombre and codigo:
                            self.programas_eafit_dict[nombre] = codigo
                
                if hasattr(self, 'msg'):
                    self._log(f"Catálogo EAFIT cargado: {len(self.programas_eafit_nombres)} programas disponibles para selección")
            else:
                if hasattr(self, 'msg'):
                    self._log("⚠️ No se encontró columna 'Nombre Programa EAFIT' en catálogo EAFIT")
        except Exception as e:
            import traceback
            traceback.print_exc()
            if hasattr(self, 'msg'):
                self._log(f"⚠️ No se pudo cargar catálogo EAFIT: {e}")
            self.programas_eafit_nombres = []
            self.programas_eafit_dict = {}
    
    def _on_cell_change(self, row_idx: int, column: str, new_val: str):
        # Guardar cambios en buffer global (persisten entre páginas)
        rows = self.table.get_rows()
        if row_idx < 0 or row_idx >= len(rows):
            return
        codigo = self._norm_codigo(rows[row_idx].get("CÓDIGO_SNIES_DEL_PROGRAMA"))
        if not codigo:
            return
        
        # Si se selecciona un programa EAFIT por nombre, actualizar el código automáticamente
        if column == "PROGRAMA_EAFIT_NOMBRE" and new_val and new_val in self.programas_eafit_dict:
            codigo_eafit = self.programas_eafit_dict[new_val]
            # Actualizar el código en la fila de datos usando set_cell_value (actualiza datos y visualización)
            if "PROGRAMA_EAFIT_CODIGO" in self.display_columns:
                self.table.set_cell_value(row_idx, "PROGRAMA_EAFIT_CODIGO", codigo_eafit)
            
            # Guardar también el código en los cambios pendientes
            if codigo not in self.pending_updates:
                self.pending_updates[codigo] = {}
            self.pending_updates[codigo]["PROGRAMA_EAFIT_CODIGO"] = codigo_eafit
            self._log(f"✓ Código EAFIT actualizado automáticamente: {codigo_eafit} para programa '{new_val[:50]}...'")
        
        # Validación inteligente: Si intenta marcar ES_REFERENTE='Sí', validar niveles
        if column == "ES_REFERENTE" and new_val.upper() in ("SÍ", "SI", "YES", "1", "TRUE"):
            nivel_programa = rows[row_idx].get("NIVEL_DE_FORMACIÓN", "")
            programa_eafit_codigo = rows[row_idx].get("PROGRAMA_EAFIT_CODIGO", "")
            programa_eafit_nombre = rows[row_idx].get("PROGRAMA_EAFIT_NOMBRE", "")
            
            # Si hay programa EAFIT asignado, validar niveles
            if programa_eafit_codigo or programa_eafit_nombre:
                nivel_valido = self._validar_niveles_coinciden(nivel_programa, programa_eafit_codigo, programa_eafit_nombre)
                if not nivel_valido[0]:
                    # Mostrar alerta
                    respuesta = messagebox.askyesno(
                        "Validación de Niveles",
                        f"⚠️ Advertencia: Los niveles de formación no coinciden.\n\n"
                        f"Programa SNIES: {nivel_programa or '(sin nivel)'}\n"
                        f"Programa EAFIT: {nivel_valido[1] or '(sin nivel)'}\n\n"
                        f"Regla del sistema: Solo se consideran referentes programas del mismo nivel.\n\n"
                        f"¿Deseas continuar de todas formas?",
                        parent=self
                    )
                    if not respuesta:
                        # Revertir cambio en la tabla
                        self.table.set_cell_value(row_idx, column, rows[row_idx].get(column, ""))
                        return
        
        upd = self.pending_updates.setdefault(codigo, {})
        upd[column] = new_val
        upd["AJUSTE_MANUAL"] = True
        upd["FECHA_AJUSTE"] = self._now_iso()
        self._touch_pending()
    
    def _validar_niveles_coinciden(self, nivel_programa: str, programa_eafit_codigo: str, programa_eafit_nombre: str) -> tuple[bool, str]:
        """
        Valida si los niveles de formación coinciden entre programa SNIES y programa EAFIT.
        
        Returns:
            (coinciden: bool, nivel_eafit: str)
        """
        from etl.clasificacionProgramas import normalizar_nivel_formacion, cargar_catalogo_eafit
        
        if not nivel_programa or pd.isna(nivel_programa):
            return (False, "")
        
        nivel_programa_norm = normalizar_nivel_formacion(str(nivel_programa))
        if not nivel_programa_norm:
            return (False, "")
        
        # Cargar catálogo EAFIT para obtener nivel del programa EAFIT
        try:
            df_catalogo = cargar_catalogo_eafit()
            
            # Buscar programa EAFIT por código o nombre
            nivel_eafit = ""
            if programa_eafit_codigo:
                # Intentar buscar por código si existe la columna
                if "Código Programa" in df_catalogo.columns:
                    mask = df_catalogo["Código Programa"] == programa_eafit_codigo
                    if mask.any():
                        idx = df_catalogo[mask].index[0]
                        nivel_eafit = df_catalogo.loc[idx, "NIVEL_DE_FORMACIÓN"] if "NIVEL_DE_FORMACIÓN" in df_catalogo.columns else ""
            
            if not nivel_eafit and programa_eafit_nombre:
                mask = df_catalogo["Nombre Programa EAFIT"] == programa_eafit_nombre
                if mask.any():
                    idx = df_catalogo[mask].index[0]
                    nivel_eafit = df_catalogo.loc[idx, "NIVEL_DE_FORMACIÓN"] if "NIVEL_DE_FORMACIÓN" in df_catalogo.columns else ""
            
            if not nivel_eafit:
                return (False, "")
            
            nivel_eafit_norm = normalizar_nivel_formacion(str(nivel_eafit))
            coinciden = nivel_programa_norm == nivel_eafit_norm
            
            return (coinciden, nivel_eafit)
        except Exception:
            # Si hay error al cargar catálogo, permitir pero advertir
            return (False, "")

    def _on_select(self, _evt=None):
        idx = self.table.get_selected_index()
        if idx is None:
            self.selection_label.config(text="Selección: -")
            return
        rows = self.table.get_rows()
        if 0 <= idx < len(rows):
            codigo = self._norm_codigo(rows[idx].get("CÓDIGO_SNIES_DEL_PROGRAMA"))
            self.selection_label.config(text=f"Selección: {codigo}")
        else:
            self.selection_label.config(text="Selección: -")

    def _pipeline_running(self) -> bool:
        lock_file = get_pipeline_lock_file()
        return lock_file.exists()

    def _poll_lock(self):
        lock_file = get_pipeline_lock_file()
        age = get_lock_age_seconds(lock_file)
        running = age is not None and age < LOCK_STALE_SECONDS
        stale = age is not None and age >= LOCK_STALE_SECONDS

        if running:
            # Deshabilitar acciones que escriben mientras el pipeline reescribe Programas.xlsx
            try:
                self.btn_save.config(state=tk.DISABLED)
                self.btn_delete.config(state=tk.DISABLED)
            except Exception:
                pass
            self.readonly_banner.config(
                text="Modo solo lectura: el pipeline está en ejecución. Espera a que termine para guardar cambios."
            )
        elif stale:
            try:
                self.btn_save.config(state=tk.DISABLED)
                self.btn_delete.config(state=tk.DISABLED)
            except Exception:
                pass
            self.readonly_banner.config(
                text="Lock antiguo detectado. Si el pipeline NO está corriendo, usa 'Desbloquear' en el Menú Principal."
            )
        else:
            try:
                self.btn_save.config(state=tk.NORMAL)
                self.btn_delete.config(state=tk.NORMAL)
            except Exception:
                pass
            self.readonly_banner.config(text="")
        # Repetir cada 1s
        try:
            root = self.winfo_toplevel()
            root.after(1000, self._poll_lock)
        except Exception:
            pass

    def _log(self, s: str):
        self.msg.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.msg.insert(tk.END, f"[{ts}] {s}\n")
        self.msg.see(tk.END)
        self.msg.config(state=tk.DISABLED)

    def _open_excel(self):
        try:
            _open_in_excel(self.file_path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)

    def _load(self):
        import pandas as pd  # Lazy import
        
        if not self.file_path.exists():
            messagebox.showerror(
                "Error",
                f"No existe {self.file_path}. Primero ejecuta el análisis SNIES.",
                parent=self,
            )
            return
        ok, msg = validate_programas_schema(self.file_path)
        if not ok:
            messagebox.showerror("Error", msg, parent=self)
            return
        try:
            if self._pipeline_running():
                self._log("⚠️ El pipeline está en ejecución. Programas.xlsx puede estar cambiando. Puedes recargar cuando termine.")
            
            # Leer todas las columnas usando función con reintentos
            from etl.exceptions_helpers import leer_excel_con_reintentos
            df_full = leer_excel_con_reintentos(self.file_path, sheet_name="Programas")
            
            # Columnas que debe tener el archivo: datos SNIES + resultado de la clasificación contra EAFIT
            required_base = [
                "CÓDIGO_SNIES_DEL_PROGRAMA",
                "NOMBRE_INSTITUCIÓN",
                "NOMBRE_DEL_PROGRAMA",
                "NIVEL_DE_FORMACIÓN",
            ]
            # Estas columnas las genera el pipeline al comparar SNIES con el catálogo EAFIT (referente sí/no y programa EAFIT asignado)
            required_classification = [
                "PROGRAMA_NUEVO",
                "ES_REFERENTE",
                "PROBABILIDAD",
                "PROGRAMA_EAFIT_CODIGO",
                "PROGRAMA_EAFIT_NOMBRE",
            ]
            missing_base = [c for c in required_base if c not in df_full.columns]
            if missing_base:
                messagebox.showerror(
                    "Error",
                    f"El archivo no tiene las columnas mínimas del SNIES:\n{', '.join(missing_base)}\n\n"
                    "Ejecuta primero el análisis SNIES (Pipeline).",
                    parent=self,
                )
                return
            
            missing_classification = [c for c in required_classification if c not in df_full.columns]
            if missing_classification:
                # PROPUESTA: Ejecutar clasificación automáticamente si faltan las columnas
                respuesta = messagebox.askyesno(
                    "Clasificación requerida",
                    "Este archivo aún no tiene la clasificación de referentes.\n\n"
                    "El sistema debe comparar cada programa del SNIES con el catálogo EAFIT (catalogoOfertasEAFIT) "
                    "y generar para cada uno: si es referente o no, y el programa EAFIT asignado (código y nombre).\n\n"
                    "¿Deseas ejecutar la clasificación ahora? (Esto puede tardar varios minutos)",
                    parent=self,
                )
                if not respuesta:
                    return
                
                # Ejecutar clasificación en hilo separado
                self._log("Ejecutando clasificación de programas nuevos...")
                self._log("Esto puede tardar varios minutos. Por favor espera...")
                
                # Obtener referencia a la ventana raíz antes de entrar al hilo
                root_window = self.winfo_toplevel()
                
                def ejecutar_clasificacion():
                    try:
                        from etl.clasificacionProgramas import clasificar_programas_nuevos
                        clasificar_programas_nuevos()
                        root_window.after(0, lambda: self._log("✓ Clasificación completada. Recargando datos..."))
                        root_window.after(0, self._load)  # Recargar después de clasificar
                    except FileNotFoundError as exc:
                        # Error específico cuando faltan modelos entrenados
                        error_msg = (
                            "No se encontraron los modelos de Machine Learning entrenados.\n\n"
                            "Para poder clasificar programas nuevos, primero debes entrenar el modelo:\n"
                            "1. Ve al menú principal\n"
                            "2. Selecciona 'Reentrenamiento del modelo'\n"
                            "3. Guarda los cambios y ejecuta el entrenamiento\n\n"
                            f"Detalle técnico: {exc}"
                        )
                        root_window.after(0, lambda: self._log(f"✗ {error_msg}"))
                        root_window.after(0, lambda: messagebox.showerror("Modelos no encontrados", error_msg, parent=self))
                    except Exception as exc:
                        error_msg = f"Error al ejecutar clasificación: {exc}"
                        root_window.after(0, lambda: self._log(f"✗ {error_msg}"))
                        root_window.after(0, lambda: messagebox.showerror("Error", error_msg, parent=self))
                
                threading.Thread(target=ejecutar_clasificacion, daemon=True).start()
                return  # Salir aquí, se recargará cuando termine la clasificación
            
            # Guardar todas las columnas disponibles
            self.all_columns = list(df_full.columns)
            # Por defecto mostrar todas las columnas (vista completa)
            self.view_complete = True
            self.display_columns = self.all_columns.copy()
            # Recrear la tabla con todas las columnas (el Treeview se creó con un subconjunto en __init__)
            self._recreate_table()
            
            # Actualizar texto del botón y habilitarlo
            self.btn_toggle_view.config(text="Vista principal", state=tk.NORMAL)

            self.df_view = df_full[self.all_columns].copy()  # Guardar todas las columnas en df_view
            self._log(f"Cargado: {self.file_path.name} ({len(self.df_view)} filas, {len(self.all_columns)} columnas disponibles).")
            # Actualizar valores del combobox de nivel con los niveles reales del archivo
            if "NIVEL_DE_FORMACIÓN" in df_full.columns:
                niveles_reales = sorted(
                    df_full["NIVEL_DE_FORMACIÓN"].dropna().astype(str).str.upper().unique().tolist()
                )
                if hasattr(self, 'nivel_filter_combo'):
                    self.nivel_filter_combo['values'] = ["TODOS"] + niveles_reales
            self._apply_filter()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo leer el Excel: {exc}", parent=self)

    def _recreate_table(self):
        """Recrea la tabla con las columnas actuales en display_columns."""
        self.table.destroy()
        # IMPORTANTE: Pasar el mismo dropdown_values_dict compartido para preservar el dropdown
        self.table = EditableTable(
            self,
            columns=self.display_columns,
            height=18,
            editable_columns=self.editable_columns,
            on_change=self._on_cell_change,
            dropdown_values=self.dropdown_values_dict,  # Pasar la referencia compartida
        )
        self.table.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.table.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _toggle_view(self):
        """Alterna entre vista completa (todas las columnas) y vista principal (9 columnas)."""
        if not self.all_columns:
            self._log("⚠️ Primero carga el archivo Programas.xlsx")
            return
        
        # Alternar estado
        self.view_complete = not self.view_complete
        
        if self.view_complete:
            # Cambiar a vista completa (todas las columnas)
            self.display_columns = self.all_columns.copy()
            self.btn_toggle_view.config(text="Vista principal")
            self._log(f"Vista completa activada ({len(self.display_columns)} columnas)")
        else:
            # Cambiar a vista principal (solo las 9 columnas principales)
            # Asegurar que las columnas principales existan en el archivo
            self.display_columns = [c for c in self.main_columns if c in self.all_columns]
            self.btn_toggle_view.config(text="Vista completa")
            self._log(f"Vista principal activada ({len(self.display_columns)} columnas)")
        
        # Recrear la tabla con las nuevas columnas
        self._recreate_table()
        
        # Reaplicar filtros con las nuevas columnas
        self._apply_filter()

    def _render_page(self):
        if self._filtered_df is None:
            self.table.set_rows([])
            self.page_label.config(text="Página: -")
            return
        total = len(self._filtered_df)
        if total == 0:
            self.table.set_rows([])
            self.page_label.config(text="Página: 0/0")
            return
        max_pages = max(1, (total + self.page_size - 1) // self.page_size)
        self.page_index = max(0, min(self.page_index, max_pages - 1))
        start = self.page_index * self.page_size
        end = min(total, start + self.page_size)
        df_page = self._filtered_df.iloc[start:end].copy()

        # Aplicar cambios pendientes a la vista (persisten entre páginas)
        if self.pending_updates:
            for i in range(len(df_page)):
                codigo = self._norm_codigo(df_page.iloc[i].get("CÓDIGO_SNIES_DEL_PROGRAMA"))
                if codigo in self.pending_updates:
                    for k, v in self.pending_updates[codigo].items():
                        if k in df_page.columns:
                            df_page.at[df_page.index[i], k] = v

        rows = df_page.to_dict(orient="records")
        self.table.set_rows(rows)
        self.page_label.config(text=f"Página: {self.page_index + 1}/{max_pages}  (filas {start + 1}-{end} de {total})")
        self._touch_pending()

    def _prev_page(self):
        if self._filtered_df is None:
            return
        if self.page_index > 0:
            self.page_index -= 1
            self._render_page()

    def _next_page(self):
        if self._filtered_df is None:
            return
        total = len(self._filtered_df)
        max_pages = max(1, (total + self.page_size - 1) // self.page_size)
        if self.page_index < (max_pages - 1):
            self.page_index += 1
            self._render_page()

    def _apply_filter(self):
        if self.df_view is None:
            return
        df = self.df_view.copy()
        mode = self.filter_var.get()
        
        # Aplicar filtro por modo
        if mode == "SOLO_NUEVOS":
            if "PROGRAMA_NUEVO" in df.columns:
                # Filtrar por programas nuevos (puede ser "Sí", "Sí ", " Sí", etc.)
                df = df[df["PROGRAMA_NUEVO"].astype(str).str.strip().str.upper() == "SÍ"]
            else:
                self._log("⚠️ Advertencia: No se encontró la columna PROGRAMA_NUEVO. Mostrando todos los programas.")
        elif mode == "SOLO_REFERENTES":
            if "ES_REFERENTE" in df.columns:
                # Filtrar por referentes (puede ser "Sí", "Sí ", " Sí", etc.)
                df = df[df["ES_REFERENTE"].astype(str).str.strip().str.upper() == "SÍ"]
            else:
                self._log("⚠️ Advertencia: No se encontró la columna ES_REFERENTE. Mostrando todos los programas.")
        # mode == "TODOS" no filtra nada

        # Aplicar filtro por nivel de formación
        nivel = getattr(self, 'nivel_filter_var', None)
        if nivel:
            nivel_sel = nivel.get().strip()
            if nivel_sel != "TODOS" and "NIVEL_DE_FORMACIÓN" in df.columns:
                df = df[
                    df["NIVEL_DE_FORMACIÓN"].astype(str)
                    .str.upper()
                    .str.contains(nivel_sel.upper(), na=False)
                ]

        # Aplicar búsqueda de texto
        q = (self.search_var.get() or "").strip().lower()
        if q:
            # buscar por código o por texto en nombre del programa/institución
            for col in ("CÓDIGO_SNIES_DEL_PROGRAMA", "NOMBRE_DEL_PROGRAMA", "NOMBRE_INSTITUCIÓN"):
                if col not in df.columns:
                    df[col] = ""
            mask = (
                df["CÓDIGO_SNIES_DEL_PROGRAMA"].astype(str).str.lower().str.contains(q, na=False)
                | df["NOMBRE_DEL_PROGRAMA"].astype(str).str.lower().str.contains(q, na=False)
                | df["NOMBRE_INSTITUCIÓN"].astype(str).str.lower().str.contains(q, na=False)
            )
            df = df[mask]

        # asegurar columnas
        for c in self.display_columns:
            if c not in df.columns:
                df[c] = ""

        df_view = df[self.display_columns].fillna("")
        self._filtered_df = df_view
        self.page_index = 0
        self._render_page()
        self._log(f"Filtro aplicado ({mode}). Total filas: {len(df_view)}")

    def _mark_si_referente(self):
        """Marca la fila seleccionada como referente (ES_REFERENTE = 'Sí')."""
        idx = self.table.get_selected_index()
        if idx is None:
            messagebox.showinfo("Info", "Selecciona una fila para marcarla.", parent=self)
            return
        rows = self.table.get_rows()
        if idx < 0 or idx >= len(rows):
            return
        row = rows[idx]
        
        # Validar niveles antes de marcar como referente
        nivel_programa = row.get("NIVEL_DE_FORMACIÓN", "")
        programa_eafit_codigo = row.get("PROGRAMA_EAFIT_CODIGO", "")
        programa_eafit_nombre = row.get("PROGRAMA_EAFIT_NOMBRE", "")
        
        # Si hay programa EAFIT asignado, validar niveles
        if programa_eafit_codigo or programa_eafit_nombre:
            nivel_valido = self._validar_niveles_coinciden(nivel_programa, programa_eafit_codigo, programa_eafit_nombre)
            if not nivel_valido[0]:
                respuesta = messagebox.askyesno(
                    "Validación de Niveles",
                    f"⚠️ Advertencia: Los niveles de formación no coinciden.\n\n"
                    f"Programa SNIES: {nivel_programa or '(sin nivel)'}\n"
                    f"Programa EAFIT: {nivel_valido[1] or '(sin nivel)'}\n\n"
                    f"Regla del sistema: Solo se consideran referentes programas del mismo nivel.\n\n"
                    f"¿Deseas marcarlo como referente de todas formas?",
                    parent=self
                )
                if not respuesta:
                    return
        
        # Si no hay programa EAFIT asignado, advertir
        if not programa_eafit_codigo and not programa_eafit_nombre:
            respuesta = messagebox.askyesno(
                "Programa EAFIT no asignado",
                "No hay un programa EAFIT asignado para este programa.\n\n"
                "¿Deseas marcarlo como referente de todas formas?\n"
                "(Puedes asignar el programa EAFIT después editando las columnas PROGRAMA_EAFIT_*)",
                parent=self
            )
            if not respuesta:
                return
        
        row["ES_REFERENTE"] = "Sí"
        codigo = self._norm_codigo(row.get("CÓDIGO_SNIES_DEL_PROGRAMA"))
        if codigo:
            upd = self.pending_updates.setdefault(codigo, {})
            upd["ES_REFERENTE"] = "Sí"
            upd["AJUSTE_MANUAL"] = True
            upd["FECHA_AJUSTE"] = self._now_iso()
            self._touch_pending()
        self.table.set_rows(rows)
        self._log("Fila marcada como SÍ referente.")
    
    def _mark_no_referente(self):
        """Marca la fila seleccionada como NO referente (ES_REFERENTE = 'No')."""
        idx = self.table.get_selected_index()
        if idx is None:
            messagebox.showinfo("Info", "Selecciona una fila para marcarla.", parent=self)
            return
        rows = self.table.get_rows()
        if idx < 0 or idx >= len(rows):
            return
        row = rows[idx]
        row["ES_REFERENTE"] = "No"
        row["PROGRAMA_EAFIT_CODIGO"] = ""
        row["PROGRAMA_EAFIT_NOMBRE"] = ""
        codigo = self._norm_codigo(row.get("CÓDIGO_SNIES_DEL_PROGRAMA"))
        if codigo:
            upd = self.pending_updates.setdefault(codigo, {})
            upd["ES_REFERENTE"] = "No"
            upd["PROGRAMA_EAFIT_CODIGO"] = ""
            upd["PROGRAMA_EAFIT_NOMBRE"] = ""
            upd["AJUSTE_MANUAL"] = True
            upd["FECHA_AJUSTE"] = self._now_iso()
            self._touch_pending()
        self.table.set_rows(rows)
        self._log("Fila marcada como NO referente (se limpió programa EAFIT asignado).")

    def _discard_row_changes(self):
        """
        Mitigación UX/negocio:
        Antes existía "Eliminar fila" (solo quitaba de la vista y podía confundir).
        Ahora esta acción descarta los cambios pendientes (buffer) de la fila seleccionada.
        """
        idx = self.table.get_selected_index()
        if idx is None:
            messagebox.showinfo("Info", "Selecciona una fila.", parent=self)
            return
        rows = self.table.get_rows()
        if not (0 <= idx < len(rows)):
            return
        codigo = self._norm_codigo(rows[idx].get("CÓDIGO_SNIES_DEL_PROGRAMA"))
        if not codigo:
            messagebox.showwarning("Atención", "La fila no tiene CÓDIGO_SNIES_DEL_PROGRAMA válido.", parent=self)
            return
        if codigo not in self.pending_updates:
            messagebox.showinfo("Info", "Esa fila no tiene cambios pendientes.", parent=self)
            return
        if not _ask_yes_no("Confirmar", "¿Descartar los cambios pendientes de esta fila?"):
            return
        try:
            self.pending_updates.pop(codigo, None)
        finally:
            self._touch_pending()
            self._render_page()
        self._log(f"Cambios pendientes descartados para: {codigo}")

    def _discard_all_changes(self):
        if not self.pending_updates:
            messagebox.showinfo("Info", "No hay cambios pendientes.", parent=self)
            return
        if not _ask_yes_no("Confirmar", "¿Descartar TODOS los cambios pendientes (todas las páginas)?"):
            return
        self.pending_updates.clear()
        self._touch_pending()
        self._render_page()
        self._log("Se descartaron todos los cambios pendientes.")

    def _save(self):
        import pandas as pd  # Lazy import
        
        if self._pipeline_running():
            messagebox.showwarning(
                "Atención",
                "El pipeline está en ejecución y está reescribiendo Programas.xlsx.\n\n"
                "Espera a que termine y vuelve a intentar guardar.",
                parent=self,
            )
            return
        if not self.pending_updates:
            messagebox.showinfo("Info", "No hay cambios pendientes para guardar.", parent=self)
            return
        self._log("Guardando cambios (todas las páginas) por CÓDIGO_SNIES_DEL_PROGRAMA...")

        # MITIGACIÓN P0 (CRÍTICA): NO perder columnas SNIES al guardar.
        # Leemos el Excel COMPLETO y aplicamos SOLO los cambios en columnas editables.
        try:
            from etl.exceptions_helpers import leer_excel_con_reintentos
            base_full = leer_excel_con_reintentos(self.file_path, sheet_name="Programas")
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo leer Programas.xlsx completo: {exc}", parent=self)
            return

        if "CÓDIGO_SNIES_DEL_PROGRAMA" not in base_full.columns:
            safe_messagebox_error("Error", "El archivo no tiene CÓDIGO_SNIES_DEL_PROGRAMA.", parent=self)
            return

        base_full["_CODIGO_NORM"] = base_full["CÓDIGO_SNIES_DEL_PROGRAMA"].apply(self._norm_codigo)

        for c in ["AJUSTE_MANUAL", "FECHA_AJUSTE"]:
            if c not in base_full.columns:
                base_full[c] = None

        for codigo, changes in self.pending_updates.items():
            mask = base_full["_CODIGO_NORM"] == codigo
            if not mask.any():
                continue
            for col, val in changes.items():
                if col in self.editable_columns or col in ("AJUSTE_MANUAL", "FECHA_AJUSTE"):
                    if col not in base_full.columns:
                        base_full[col] = None
                    
                    # Convertir valores al tipo correcto según la columna
                    try:
                        if col == "PROGRAMA_EAFIT_CODIGO":
                            # Convertir a numérico (int o float según el tipo original)
                            if val == "" or val is None or pd.isna(val):
                                val_converted = None
                            else:
                                # Intentar convertir a int primero, luego float si falla
                                try:
                                    val_converted = int(float(str(val)))
                                except (ValueError, TypeError):
                                    val_converted = None
                            # Convertir la columna a object si es necesario para permitir valores mixtos
                            if base_full[col].dtype != 'object':
                                base_full[col] = base_full[col].astype('object')
                            base_full.loc[mask, col] = val_converted
                        elif col == "PROBABILIDAD":
                            # Convertir a float
                            if val == "" or val is None or pd.isna(val):
                                val_converted = None
                            else:
                                try:
                                    val_converted = float(str(val))
                                except (ValueError, TypeError):
                                    val_converted = None
                            base_full.loc[mask, col] = val_converted
                        else:
                            # Para otras columnas (ES_REFERENTE, PROGRAMA_EAFIT_NOMBRE, etc.), mantener como string
                            base_full.loc[mask, col] = val
                    except Exception as e:
                        # Si hay error al convertir, intentar asignar directamente y convertir la columna a object
                        try:
                            if base_full[col].dtype != 'object':
                                base_full[col] = base_full[col].astype('object')
                            base_full.loc[mask, col] = val
                        except Exception as e2:
                            self._log(f"⚠️ Error al guardar {col} para código {codigo}: {e2}")
                            continue

        try:
            # Usar mode="w" para sobrescribir completamente el archivo (más seguro que mode="a")
            with pd.ExcelWriter(self.file_path, mode="w", engine="openpyxl") as writer:
                base_full.drop(columns=["_CODIGO_NORM"]).to_excel(writer, sheet_name="Programas", index=False)

            # Intentar retro-sincronizar el histórico con los ajustes manuales
            try:
                from etl.historicoProgramasNuevos import sincronizar_historico_con_ajustes_manuales
                sincronizar_historico_con_ajustes_manuales()
                self._log("✓ Histórico sincronizado con ajustes manuales en HistoricoProgramasNuevos .xlsx")
                sincronizado_ok = True
            except Exception as exc:
                sincronizado_ok = False
                # No fallar si no se puede actualizar el histórico, solo registrar advertencia
                self._log(f"⚠️ No se pudo sincronizar el histórico con los ajustes manuales: {exc}")
                try:
                    from etl.pipeline_logger import log_warning
                    log_warning(f"Error al sincronizar histórico con ajustes manuales: {exc}")
                except Exception:
                    # Si no está disponible, solo usamos el log de la UI
                    pass

            self.pending_updates.clear()
            self._touch_pending()
            self._load()
            self._log("✓ Cambios guardados en Programas.xlsx")
            mensaje_hist = " y en HistoricoProgramasNuevos .xlsx" if sincronizado_ok else ""
            messagebox.showinfo(
                "Guardado",
                f"Los cambios se guardaron correctamente en Programas.xlsx{mensaje_hist}.",
                parent=self
            )
        except PermissionError:
            safe_messagebox_error("Error", explain_file_in_use(), parent=self)
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo guardar: {exc}", parent=self)
    
    def _restore_backup(self):
        """Restaura el estado anterior desde el backup oculto."""
        if not self.last_backup_path or not self.last_backup_path.exists():
            messagebox.showwarning(
                "No hay backup",
                "No hay un backup disponible para restaurar.\n\n"
                "El backup solo está disponible después de guardar cambios.",
                parent=self
            )
            return
        
        if not _ask_yes_no(
            "Confirmar Restauración",
            f"¿Restaurar el estado anterior desde {self.last_backup_path.name}?\n\n"
            "Esto sobrescribirá todos los cambios guardados en Programas.xlsx.\n"
            "Los cambios pendientes (no guardados) se perderán.",
            parent=self
        ):
            return
        
        import pandas as pd
        
        try:
            # Leer backup usando función con reintentos
            from etl.exceptions_helpers import leer_excel_con_reintentos
            df_backup = leer_excel_con_reintentos(self.last_backup_path, sheet_name="Programas")
            
            # Escribir sobre Programas.xlsx
            with pd.ExcelWriter(self.file_path, mode="w", engine="openpyxl") as writer:
                df_backup.to_excel(writer, sheet_name="Programas", index=False)
            
            # Limpiar cambios pendientes
            self.pending_updates.clear()
            self._touch_pending()
            
            # Recargar
            self._load()
            
            # Deshabilitar botón de restaurar
            self.btn_restore.config(state=tk.DISABLED)
            self.last_backup_path = None
            
            self._log("✓ Estado anterior restaurado desde backup")
            messagebox.showinfo(
                "Restaurado",
                "El estado anterior se restauró correctamente desde el backup.",
                parent=self
            )
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo restaurar: {exc}", parent=self)


class RetrainPage(ttk.Frame):
    """Edición de referentes + reentrenamiento del modelo."""

    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        
        # Import lazy de pandas y módulos ETL (solo cuando se abre esta página)
        import pandas as pd
        from etl.config import get_archivo_referentes, leer_datos_flexible, REF_DIR
        
        self.base_dir = ensure_base_dir(self)
        if not self.base_dir:
            if on_back:
                on_back()
            return

        self._get_referentes = get_archivo_referentes
        self._leer = leer_datos_flexible
        self.file_path = self._get_referentes()
        self.df = None  # type: ignore

        header = ttk.Frame(self, padding=12, style="Page.TFrame")
        header.pack(fill=tk.X)
        header_left = ttk.Frame(header, style="Page.TFrame")
        header_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(header_left, text="🎯 Reentrenamiento del modelo", style="Header.TLabel").pack(anchor="w")
        self.subheader_label = ttk.Label(header_left, text="Edita referentesUnificados (ref/). Guarda cambios y luego reentrena para que el modelo use los nuevos referentes.", style="SubHeader.TLabel")
        self.subheader_label.pack(anchor="w", pady=(4, 0), fill=tk.X)
        if on_back:
            ttk.Button(header, text="← Volver al menú", command=lambda: on_back() if on_back else None, style="Back.TButton").pack(side=tk.RIGHT)

        # Botones en grid para mejor organización responsive
        btns = ttk.Frame(self, padding="10")
        btns.pack(fill=tk.X)
        btns.grid_columnconfigure(0, weight=1, uniform="btn_col")
        btns.grid_columnconfigure(1, weight=1, uniform="btn_col")
        
        # Fila 1: Archivo y edición
        row1 = ttk.Frame(btns, style="App.TFrame")
        row1.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(row1, text="Cargar archivo de referentes", command=self._load).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row1, text="Abrir en Excel", command=self._open_excel).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Agregar fila", command=self._add_row).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Eliminar fila", command=self._delete_row).pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Guardar cambios", command=self._save).pack(side=tk.LEFT, padx=6)
        
        # Fila 2: Sincronización y entrenamiento
        row2 = ttk.Frame(btns, style="App.TFrame")
        row2.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(row2, text="Sincronizar ajustes manuales", command=self._sync_manual_adjustments).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row2, text="Simular reentrenamiento", command=self._dry_run_train).pack(side=tk.LEFT, padx=6)
        ttk.Button(row2, text="Reentrenar modelo", command=self._train).pack(side=tk.LEFT, padx=6)
        
        # Selector de versión y rollback
        version_frame = ttk.Frame(self, padding="10")
        version_frame.pack(fill=tk.X)
        ttk.Label(version_frame, text="Versión del modelo:").pack(side=tk.LEFT, padx=(0, 6))
        self.version_var = tk.StringVar(value="actual")
        self.version_combo = ttk.Combobox(
            version_frame,
            textvariable=self.version_var,
            state="readonly",
            width=15
        )
        self.version_combo.pack(side=tk.LEFT, padx=6)
        # NO llamar _update_version_list() aquí porque self.msg aún no existe
        # Se llamará después de crear self.msg
        ttk.Button(version_frame, text="Usar esta versión", command=self._switch_version).pack(side=tk.LEFT, padx=6)
        ttk.Button(version_frame, text="Rollback a versión anterior", command=self._rollback_version).pack(side=tk.LEFT, padx=6)

        self.msg = tk.Text(self, height=7, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9), bg=EAFIT["card_bg"], fg=EAFIT["text"])
        self.msg.pack(fill=tk.X, padx=10, pady=(0, 10))

        cols = [
            "NOMBRE_DEL_PROGRAMA",
            "NombrePrograma EAFIT",
            "CAMPO_AMPLIO",
            "CAMPO_AMPLIO_EAFIT",
            "NIVEL_DE_FORMACIÓN",
            "NIVEL_DE_FORMACIÓN EAFIT",
            "label",
        ]
        self.table = EditableTable(self, columns=cols, height=20)
        self.table.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._log(f"Archivo actual detectado: {self.file_path}")
        self._log("Tip: mantén label=1 para referentes confirmados (es lo que usa el entrenamiento).")
        self._update_version_list()

    def _on_resize(self, w: int, h: int) -> None:
        """Responsive: ajusta altura de tabla, wraplengths y log."""
        try:
            table_pixels = max(120, h - 380)
            self.table.set_height_from_pixels(table_pixels)
        except (tk.TclError, AttributeError):
            pass
        if hasattr(self, "subheader_label"):
            try:
                self.subheader_label.config(wraplength=max(400, w - 100))
            except (tk.TclError, AttributeError):
                pass
        try:
            log_lines = max(5, min(12, (h - 380) // 20))
            self.msg.config(height=log_lines)
        except (tk.TclError, AttributeError):
            pass
    
    def _update_version_list(self):
        """Actualiza la lista de versiones disponibles en el combobox."""
        try:
            from etl.clasificacionProgramas import listar_versiones_modelos
            
            versiones = listar_versiones_modelos()
            valores = ["actual"] + [f"v{v}" for v in versiones]
            self.version_combo['values'] = valores
            
            if versiones:
                self._log(f"Versiones disponibles: actual, {', '.join([f'v{v}' for v in versiones])}")
            else:
                self._log("No hay versiones guardadas. El próximo entrenamiento creará v1.")
        except Exception as e:
            self._log(f"Error al listar versiones: {e}")
    
    def _switch_version(self):
        """Cambia la versión actual del modelo."""
        version_str = self.version_var.get()
        
        if version_str == "actual":
            messagebox.showinfo("Info", "Ya estás usando la versión actual.", parent=self)
            return
        
        try:
            version_num = int(version_str.replace("v", ""))
            from etl.clasificacionProgramas import obtener_rutas_modelo_version, MODELS_DIR
            import shutil
            
            ruta_clasificador, ruta_embeddings, ruta_encoder = obtener_rutas_modelo_version(version_num)
            
            # Verificar que existan
            if not all([ruta_clasificador.exists(), ruta_embeddings.exists(), ruta_encoder.exists()]):
                messagebox.showerror("Error", f"La versión {version_str} no existe completamente.", parent=self)
                return
            
            # Hacer backup de versión actual si existe
            from etl.clasificacionProgramas import MODELO_CLASIFICADOR, MODELO_EMBEDDINGS_OBJ, ENCODER_PROGRAMAS_EAFIT
            if MODELO_CLASIFICADOR.exists():
                backup_version = version_num - 1 if version_num > 1 else 1
                ruta_backup_clasificador, ruta_backup_embeddings, ruta_backup_encoder = obtener_rutas_modelo_version(backup_version)
                try:
                    shutil.copy2(MODELO_CLASIFICADOR, ruta_backup_clasificador)
                    shutil.copy2(MODELO_EMBEDDINGS_OBJ, ruta_backup_embeddings)
                    shutil.copy2(ENCODER_PROGRAMAS_EAFIT, ruta_backup_encoder)
                except Exception:
                    pass
            
            # Copiar versión seleccionada a versión actual
            shutil.copy2(ruta_clasificador, MODELO_CLASIFICADOR)
            shutil.copy2(ruta_embeddings, MODELO_EMBEDDINGS_OBJ)
            shutil.copy2(ruta_encoder, ENCODER_PROGRAMAS_EAFIT)
            
            self._log(f"✓ Versión {version_str} establecida como versión actual")
            messagebox.showinfo("Versión cambiada", f"La versión {version_str} ahora es la versión actual.", parent=self)
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo cambiar la versión: {exc}", parent=self)
    
    def _rollback_version(self):
        """Hace rollback a la versión anterior."""
        try:
            from etl.clasificacionProgramas import listar_versiones_modelos, obtener_rutas_modelo_version, MODELO_CLASIFICADOR
            import shutil
            
            versiones = listar_versiones_modelos()
            if not versiones or len(versiones) < 2:
                messagebox.showwarning(
                    "No hay versión anterior",
                    "No hay suficientes versiones para hacer rollback.\n\n"
                    "Necesitas al menos 2 versiones guardadas.",
                    parent=self
                )
                return
            
            # Obtener versión anterior (la penúltima)
            version_anterior = versiones[-2]
            
            if not _ask_yes_no(
                "Confirmar Rollback",
                f"¿Hacer rollback a la versión v{version_anterior}?\n\n"
                f"Esto sobrescribirá la versión actual con v{version_anterior}.",
                parent=self
            ):
                return
            
            ruta_clasificador, ruta_embeddings, ruta_encoder = obtener_rutas_modelo_version(version_anterior)
            
            # Verificar que existan
            if not all([ruta_clasificador.exists(), ruta_embeddings.exists(), ruta_encoder.exists()]):
                messagebox.showerror("Error", f"La versión v{version_anterior} no existe completamente.", parent=self)
                return
            
            # Copiar versión anterior a versión actual
            from etl.clasificacionProgramas import MODELO_CLASIFICADOR, MODELO_EMBEDDINGS_OBJ, ENCODER_PROGRAMAS_EAFIT
            shutil.copy2(ruta_clasificador, MODELO_CLASIFICADOR)
            shutil.copy2(ruta_embeddings, MODELO_EMBEDDINGS_OBJ)
            shutil.copy2(ruta_encoder, ENCODER_PROGRAMAS_EAFIT)
            
            self._log(f"✓ Rollback completado: versión actual ahora es v{version_anterior}")
            messagebox.showinfo("Rollback completado", f"La versión actual ahora es v{version_anterior}.", parent=self)
            self._update_version_list()
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo hacer rollback: {exc}", parent=self)
    
    def _dry_run_train(self):
        """Simula el reentrenamiento y muestra métricas estimadas."""
        if self.df is None:
            messagebox.showwarning("Atención", "Primero carga el archivo de referentes.", parent=self)
            return
        
        self._log("Simulando reentrenamiento...")
        
        def run():
            try:
                import pandas as pd
                from etl.clasificacionProgramas import (
                    cargar_referentes,
                    entrenar_modelo,
                    preparar_features_entrenamiento,
                    _get_sentence_transformer
                )

                # Cargar datos usando cargar_referentes() que aplica
                # la normalización que necesita preparar_features_entrenamiento
                df_actual = cargar_referentes(self.file_path)

                if df_actual is None or len(df_actual) == 0:
                    self.after(0, lambda: safe_messagebox_error(
                        "Error", "cargar_referentes() no devolvió datos válidos.", parent=self
                    ))
                    return

                # Validar mínimos sobre el DataFrame ya normalizado
                ok, msg = self._validate_referentes(df_actual.copy())
                if not ok:
                    self.after(0, lambda: safe_messagebox_error("Error", f"No se puede simular: {msg}", parent=self))
                    return

                # Cargar modelo de embeddings
                SentenceTransformer = _get_sentence_transformer()
                modelo_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

                # Preparar features (df_actual ya tiene las columnas _norm necesarias)
                features, labels, encoder = preparar_features_entrenamiento(df_actual, modelo_embeddings)
                
                # Entrenar modelo temporal
                modelo_temp, metricas = entrenar_modelo(features, labels)
                
                # Comparar con modelo actual si existe
                accuracy_actual = None
                try:
                    from etl.clasificacionProgramas import cargar_modelos
                    modelo_actual, _, _ = cargar_modelos()
                    # Evaluar modelo actual con mismo test set
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
                    accuracy_actual = modelo_actual.score(X_test, y_test)
                except Exception:
                    pass
                
                # Mostrar resultados
                accuracy_nuevo = metricas['accuracy']
                
                mensaje = f"Simulación completada:\n\n"
                if accuracy_actual is not None:
                    mensaje += f"Precisión actual: {accuracy_actual:.2%}\n"
                    mensaje += f"Precisión nueva (estimada): {accuracy_nuevo:.2%}\n\n"
                    diferencia = accuracy_nuevo - accuracy_actual
                    if diferencia > 0:
                        mensaje += f"✅ Mejora: +{diferencia:.2%}"
                    elif diferencia < 0:
                        mensaje += f"⚠️ Empeora: {diferencia:.2%}"
                    else:
                        mensaje += "➡️ Sin cambios"
                else:
                    mensaje += f"Precisión nueva (estimada): {accuracy_nuevo:.2%}\n\n"
                    mensaje += "(No hay modelo actual para comparar)"
                
                self.after(0, lambda: self._log(f"Simulación: {mensaje}"))
                self.after(0, lambda: messagebox.showinfo("Simulación de Reentrenamiento", mensaje, parent=self))
            except Exception as exc:
                self.after(0, lambda: self._log(f"✗ Error en simulación: {exc}"))
                self.after(0, lambda: safe_messagebox_error("Error", f"No se pudo simular: {exc}", parent=self))
        
        threading.Thread(target=run, daemon=True).start()

    def _log(self, s: str):
        # Verificar que self.msg existe antes de usarlo (para evitar errores durante inicialización)
        if not hasattr(self, 'msg') or self.msg is None:
            return
        try:
            self.msg.config(state=tk.NORMAL)
            ts = time.strftime("%H:%M:%S")
            self.msg.insert(tk.END, f"[{ts}] {s}\n")
            self.msg.see(tk.END)
            self.msg.config(state=tk.DISABLED)
        except (tk.TclError, AttributeError):
            # Si el widget fue destruido o no está disponible, ignorar silenciosamente
            pass

    def _open_excel(self):
        try:
            _open_in_excel(self.file_path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)

    def _load(self):
        if not self.file_path.exists():
            messagebox.showerror("Error", f"No existe el archivo: {self.file_path}", parent=self)
            return
        try:
            self.df = self._leer(self.file_path)
            self._log(f"Cargado: {self.file_path.name} ({len(self.df)} filas)")
            # asegurar columnas
            for c in self.table.columns:
                if c not in self.df.columns:
                    self.df[c] = ""
            df_view = self.df[self.table.columns].fillna("")
            self.table.set_rows(df_view.to_dict(orient="records"))
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar: {exc}", parent=self)

    def _add_row(self):
        self.table.add_row({"label": 1})

    def _delete_row(self):
        if not _ask_yes_no("Confirmar", "¿Eliminar la fila seleccionada? (No guarda hasta Guardar cambios)"):
            return
        self.table.delete_selected()

    def _validate_referentes(self, df_out) -> tuple[bool, str]:
        import pandas as pd  # Lazy import
        required = [
            "NOMBRE_DEL_PROGRAMA",
            "NombrePrograma EAFIT",
            "label",
        ]
        missing = [c for c in required if c not in df_out.columns]
        if missing:
            return False, f"Faltan columnas requeridas: {', '.join(missing)}"

        # Normalizar label
        try:
            df_out["label"] = pd.to_numeric(df_out["label"], errors="coerce").fillna(0).astype(int)
        except Exception:
            return False, "La columna 'label' debe ser numérica (0/1)."

        n_pos = int((df_out["label"] == 1).sum())
        if n_pos == 0:
            return False, "No hay registros con label=1. El entrenamiento quedaría sin referentes confirmados."

        # Nombres mínimos
        if df_out["NOMBRE_DEL_PROGRAMA"].astype(str).str.strip().eq("").all():
            return False, "NOMBRE_DEL_PROGRAMA está vacío en todas las filas."
        if df_out["NombrePrograma EAFIT"].astype(str).str.strip().eq("").all():
            return False, "NombrePrograma EAFIT está vacío en todas las filas."

        return True, f"OK (label=1: {n_pos})"

    def _save(self):
        rows = self.table.get_rows()
        if not rows:
            messagebox.showwarning("Atención", "No hay filas para guardar.", parent=self)
            return

        import pandas as pd
        df_gui = pd.DataFrame(rows)

        # Validar antes de tocar el disco
        ok, msg = self._validate_referentes(df_gui.copy())
        if not ok:
            safe_messagebox_error("Error", msg, parent=self)
            return
        self._log(f"Validación de referentes: {msg}")

        # Leer el archivo COMPLETO para preservar todas las columnas
        try:
            df_completo = self._leer(self.file_path)
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo leer el archivo para actualizar: {exc}", parent=self)
            return

        # Backup antes de guardar
        try:
            import shutil, time
            backup = self.file_path.parent / f"{self.file_path.stem}__backup_{time.strftime('%Y%m%d_%H%M%S')}{self.file_path.suffix}"
            shutil.copy2(self.file_path, backup)
            self._log(f"Backup creado: {backup.name}")
        except Exception:
            pass

        # Actualizar columnas editables en el archivo completo usando las filas de la GUI
        # Clave de match: NOMBRE_DEL_PROGRAMA + NombrePrograma EAFIT
        cols_editables = [c for c in self.table.columns if c in df_completo.columns]
        df_completo = df_completo.reset_index(drop=True)
        df_gui = df_gui.reset_index(drop=True)

        # Si tienen el mismo número de filas y mismo orden, actualizar directamente
        if len(df_gui) == len(df_completo):
            for col in cols_editables:
                if col in df_gui.columns:
                    df_completo[col] = df_gui[col].values
        else:
            # Si difieren (el usuario agregó o borró filas en la GUI), hacer merge por clave
            key_cols = ['NOMBRE_DEL_PROGRAMA', 'NombrePrograma EAFIT']
            if all(c in df_completo.columns and c in df_gui.columns for c in key_cols):
                df_completo = df_completo.set_index(key_cols)
                df_gui_indexed = df_gui[key_cols + [c for c in cols_editables if c in df_gui.columns]].set_index(key_cols)
                df_completo.update(df_gui_indexed)
                df_completo = df_completo.reset_index()
            else:
                # PROTECCIÓN CRÍTICA: si las llaves no coinciden, bloquear el guardado.
                # df_gui solo tiene las columnas visibles — sobrescribir sería pérdida de datos.
                safe_messagebox_error(
                    "Error de integridad",
                    "No se encontró la clave de emparejamiento (NOMBRE_DEL_PROGRAMA + "
                    "NombrePrograma EAFIT) entre la tabla y el archivo en disco.\n\n"
                    "No se guardaron cambios para proteger la integridad del archivo de referentes.\n\n"
                    "Recarga el archivo y vuelve a intentar.",
                    parent=self,
                )
                self._log("✗ Guardado cancelado: clave de match no encontrada. Archivo sin modificar.")
                return

        try:
            if self.file_path.suffix.lower() == ".csv":
                df_completo.to_csv(self.file_path, index=False, encoding="utf-8")
            else:
                with pd.ExcelWriter(self.file_path, mode="w", engine="openpyxl") as writer:
                    df_completo.to_excel(writer, index=False)
            self._log("Cambios guardados preservando todas las columnas del archivo.")
            messagebox.showinfo("OK", "Cambios guardados.", parent=self)
        except PermissionError:
            safe_messagebox_error("Error", explain_file_in_use(), parent=self)
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo guardar: {exc}", parent=self)

    def _train(self):
        import pandas as pd  # Lazy import
        
        if not _ask_yes_no("Confirmar", "¿Reentrenar el modelo ahora? Esto puede tardar varios minutos."):
            return
        # Validar archivo antes de entrenar
        try:
            df_tmp = self._leer(self.file_path)
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo leer el archivo de referentes: {exc}", parent=self)
            return
        for c in self.table.columns:
            if c not in df_tmp.columns:
                df_tmp[c] = ""
        df_tmp = df_tmp[self.table.columns].fillna("")
        ok, msg = self._validate_referentes(df_tmp.copy())
        if not ok:
            safe_messagebox_error("Error", f"No se puede reentrenar: {msg}", parent=self)
            return
        self._log(f"Validación previa a entrenamiento: {msg}")
        self._log("Iniciando reentrenamiento...")

        def run():
            try:
                from etl.clasificacionProgramas import entrenar_y_guardar_modelo

                entrenar_y_guardar_modelo()
                self.after(0, lambda: self._log("✓ Reentrenamiento completado"))
                self.after(0, lambda: messagebox.showinfo("OK", "Modelo reentrenado y guardado en models/.", parent=self))
            except Exception as exc:
                self.after(0, lambda: self._log(f"✗ Error reentrenando: {exc}"))
                self.after(0, lambda: messagebox.showerror("Error", f"No se pudo reentrenar: {exc}", parent=self))

        threading.Thread(target=run, daemon=True).start()
    
    def _sync_manual_adjustments(self):
        """
        Sincroniza los ajustes manuales de Programas.xlsx con referentesUnificados.csv.
        
        Esto previene que falsos positivos corregidos manualmente entrenen el modelo.
        """
        import pandas as pd
        from etl.normalizacion import ARCHIVO_PROGRAMAS
        from etl.config import get_archivo_referentes
        from etl.exceptions_helpers import leer_excel_con_reintentos
        
        if not _ask_yes_no(
            "Sincronizar Ajustes Manuales",
            "Esta función sincronizará los ajustes manuales de Programas.xlsx con referentesUnificados.csv.\n\n"
            "Acciones que se realizarán:\n"
            "1. Programas marcados como ES_REFERENTE='No' con AJUSTE_MANUAL=True → se eliminarán del archivo de referentes (falsos positivos)\n"
            "2. Programas marcados como ES_REFERENTE='Sí' con AJUSTE_MANUAL=True → se agregarán como referentes si no existen\n\n"
            "¿Deseas continuar?",
            parent=self
        ):
            return
        
        self._log("=== Sincronizando ajustes manuales ===")
        
        try:
            # 1. Leer Programas.xlsx y filtrar ajustes manuales
            self._log("Leyendo ajustes manuales desde Programas.xlsx...")
            df_programas = leer_excel_con_reintentos(ARCHIVO_PROGRAMAS, sheet_name="Programas")
            
            # Normalizar código SNIES
            def norm_codigo(v):
                s = str(v).strip() if v is not None else ""
                return s[:-2] if s.endswith(".0") else s
            
            df_programas["_CODIGO_NORM"] = df_programas["CÓDIGO_SNIES_DEL_PROGRAMA"].apply(norm_codigo)
            
            # Filtrar solo ajustes manuales
            if "AJUSTE_MANUAL" not in df_programas.columns:
                self._log("⚠️ No se encontró columna AJUSTE_MANUAL. No hay ajustes para sincronizar.")
                messagebox.showinfo("Info", "No se encontraron ajustes manuales para sincronizar.", parent=self)
                return
            
            # Convertir AJUSTE_MANUAL a bool
            def _to_bool(v):
                if v is None:
                    return False
                if isinstance(v, bool):
                    return v
                s = str(v).strip().lower()
                return s in ("1", "true", "t", "yes", "y", "si", "sí")
            
            df_programas["AJUSTE_MANUAL"] = df_programas["AJUSTE_MANUAL"].apply(_to_bool)
            df_ajustes = df_programas[df_programas["AJUSTE_MANUAL"] == True].copy()
            
            if len(df_ajustes) == 0:
                self._log("No hay ajustes manuales para sincronizar.")
                messagebox.showinfo("Info", "No se encontraron ajustes manuales para sincronizar.", parent=self)
                return
            
            self._log(f"Encontrados {len(df_ajustes)} ajustes manuales")
            
            # 2. Leer referentesUnificados
            archivo_referentes = get_archivo_referentes()
            if not archivo_referentes.exists():
                safe_messagebox_error("Error", f"No se encontró el archivo de referentes: {archivo_referentes}", parent=self)
                return
            
            self._log(f"Leyendo referentes desde {archivo_referentes.name}...")
            df_referentes = self._leer(archivo_referentes)
            
            # Normalizar código en referentes
            if "CÓDIGO_SNIES_DEL_PROGRAMA" in df_referentes.columns:
                df_referentes["_CODIGO_NORM"] = df_referentes["CÓDIGO_SNIES_DEL_PROGRAMA"].apply(norm_codigo)
            else:
                df_referentes["_CODIGO_NORM"] = ""
            
            # Asegurar que existe columna label
            if "label" not in df_referentes.columns:
                df_referentes["label"] = 1
            
            # Normalizar label
            df_referentes["label"] = pd.to_numeric(df_referentes["label"], errors="coerce").fillna(0).astype(int)

            # Separar ajustes: los que desmarcan (falsos positivos) y los que confirman
            codigos_falsos_positivos = set()
            codigos_nuevos_referentes = []

            for idx, row_ajuste in df_ajustes.iterrows():
                codigo = row_ajuste["_CODIGO_NORM"]
                if not codigo:
                    continue
                es_referente = str(row_ajuste.get("ES_REFERENTE", "")).strip().upper() in (
                    "SÍ", "SI", "YES", "1", "TRUE"
                )
                if es_referente:
                    codigos_nuevos_referentes.append(row_ajuste)
                else:
                    codigos_falsos_positivos.add(codigo)

            registros_actualizados = 0

            # Eliminar falsos positivos del referente (en vez de poner label=0 que no tiene efecto)
            if codigos_falsos_positivos:
                mask_eliminar = df_referentes["_CODIGO_NORM"].isin(codigos_falsos_positivos)
                n_antes = len(df_referentes)
                df_referentes = df_referentes[~mask_eliminar].copy()
                eliminados = n_antes - len(df_referentes)
                if eliminados > 0:
                    registros_actualizados += eliminados
                    self._log(f"Eliminados {eliminados} falsos positivos del referente de entrenamiento")

            # Agregar nuevos referentes confirmados si no existen ya
            for row_ajuste in codigos_nuevos_referentes:
                codigo = row_ajuste["_CODIGO_NORM"]
                if codigo in set(df_referentes["_CODIGO_NORM"]):
                    continue  # Ya existe, no duplicar
                nuevo_referente = {
                    "CÓDIGO_SNIES_DEL_PROGRAMA": codigo,
                    "NOMBRE_DEL_PROGRAMA": str(row_ajuste.get("NOMBRE_DEL_PROGRAMA", "")),
                    "NombrePrograma EAFIT": str(row_ajuste.get("PROGRAMA_EAFIT_NOMBRE", "")),
                    "CAMPO_AMPLIO": str(row_ajuste.get("CINE_F_2013_AC_CAMPO_AMPLIO", "")),
                    "CAMPO_AMPLIO_EAFIT": "",
                    "NIVEL_DE_FORMACIÓN": str(row_ajuste.get("NIVEL_DE_FORMACIÓN", "")),
                    "NIVEL_DE_FORMACIÓN EAFIT": str(row_ajuste.get("NIVEL_DE_FORMACIÓN", "")),
                    "label": 1,
                    "_CODIGO_NORM": codigo,
                }
                for col in df_referentes.columns:
                    if col not in nuevo_referente:
                        nuevo_referente[col] = ""
                df_referentes = pd.concat(
                    [df_referentes, pd.DataFrame([nuevo_referente])], ignore_index=True
                )
                registros_actualizados += 1
                self._log(f"Agregado nuevo referente confirmado: {codigo}")

            # 4. Guardar referentes actualizados
            if registros_actualizados > 0:
                # Eliminar columna temporal
                if "_CODIGO_NORM" in df_referentes.columns:
                    df_referentes = df_referentes.drop(columns=["_CODIGO_NORM"])
                
                # Backup antes de guardar
                try:
                    backup = archivo_referentes.parent / f"{archivo_referentes.stem}__backup_sync_{time.strftime('%Y%m%d_%H%M%S')}{archivo_referentes.suffix}"
                    shutil.copy2(archivo_referentes, backup)
                    self._log(f"Backup creado: {backup.name}")
                except Exception as e:
                    self._log(f"Advertencia: No se pudo crear backup: {e}")
                
                # Guardar
                if archivo_referentes.suffix.lower() == ".csv":
                    df_referentes.to_csv(archivo_referentes, index=False, encoding="utf-8")
                else:
                    with pd.ExcelWriter(archivo_referentes, mode="w", engine="openpyxl") as writer:
                        df_referentes.to_excel(writer, index=False)
                
                self._log(f"✓ Sincronización completada (cambios aplicados: {registros_actualizados})")

                messagebox.showinfo(
                    "Sincronización Completada",
                    f"Sincronización exitosa:\n\n"
                    f"Operaciones aplicadas: {registros_actualizados}\n\n"
                    f"Los falsos positivos fueron eliminados del archivo de referentes y ya no entrenan el modelo.",
                    parent=self
                )
                
                # Recargar tabla
                self._load()
            else:
                self._log("No se realizaron cambios (los ajustes ya estaban sincronizados)")
                messagebox.showinfo("Info", "No se realizaron cambios. Los ajustes ya estaban sincronizados.", parent=self)
                
        except Exception as exc:
            error_msg = f"Error al sincronizar: {exc}"
            self._log(f"✗ {error_msg}")
            safe_messagebox_error("Error", error_msg, parent=self)

    def _on_resize(self, w: int, h: int) -> None:
        """Responsive: ajusta la altura de la tabla y wraplengths al espacio disponible."""
        table_pixels = max(120, h - 340)
        self.table.set_height_from_pixels(table_pixels)
        
        # Ajustar wraplength del subheader dinámicamente
        if hasattr(self, 'subheader_label'):
            wraplength = max(400, w - 100)
            self.subheader_label.config(wraplength=wraplength)


class MergePage(ttk.Frame):
    """Consolidación (merge) de Programas.xlsx con un archivo histórico."""

    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        
        # Import lazy de pandas y módulos ETL (solo cuando se abre esta página)
        import pandas as pd
        from etl.config import ARCHIVO_HISTORICO, OUTPUTS_DIR
        from etl.normalizacion import ARCHIVO_PROGRAMAS
        
        self.base_dir = ensure_base_dir(self)
        if not self.base_dir:
            if on_back:
                on_back()
            return

        self.default_current = ARCHIVO_PROGRAMAS
        self.default_hist = ARCHIVO_HISTORICO
        self.outputs_dir = OUTPUTS_DIR

        frame = ttk.Frame(self, padding=14, style="Page.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(frame, style="Page.TFrame")
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(header_frame, text="🔀 Consolidar archivos (Merge)", style="Header.TLabel").pack(side=tk.LEFT)
        if on_back:
            ttk.Button(header_frame, text="← Volver al menú", command=lambda: on_back() if on_back else None, style="Back.TButton").pack(side=tk.RIGHT)
        self.subheader_label = ttk.Label(frame, text="Combina el archivo actual con un histórico. Los registros con ajuste manual tienen prioridad.", style="SubHeader.TLabel")
        self.subheader_label.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 12))

        ttk.Label(frame, text="Archivo actual (Programas.xlsx):").grid(row=2, column=0, sticky="w")
        self.current_var = tk.StringVar(value=str(self.default_current))
        self.entry_current = ttk.Entry(frame, textvariable=self.current_var, width=80)
        self.entry_current.grid(row=2, column=1, sticky="ew")
        ttk.Button(frame, text="Buscar...", command=self._pick_current).grid(row=2, column=2, padx=6)

        ttk.Label(frame, text="Archivo histórico:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.hist_var = tk.StringVar(value=str(self.default_hist))
        self.entry_hist = ttk.Entry(frame, textvariable=self.hist_var, width=80)
        self.entry_hist.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(frame, text="Buscar...", command=self._pick_hist).grid(row=3, column=2, padx=6, pady=(8, 0))

        ttk.Label(frame, text="Archivo de salida:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.out_var = tk.StringVar(value=str(self.outputs_dir / "ProgramasConsolidado.xlsx"))
        self.entry_out = ttk.Entry(frame, textvariable=self.out_var, width=80)
        self.entry_out.grid(row=4, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(frame, text="Elegir...", command=self._pick_out).grid(row=4, column=2, padx=6, pady=(8, 0))

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=5, column=0, columnspan=3, sticky="w", pady=(14, 0))
        ttk.Button(btn_row, text="Consolidar", command=self._merge, style="Primary.TButton").pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Abrir salida", command=self._open_out, style="Secondary.TButton").pack(side=tk.LEFT, padx=8)

        self.msg = tk.Text(frame, height=8, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9), bg=EAFIT["card_bg"], fg=EAFIT["text"])
        self.msg.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(12, 0))

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(6, weight=1)

    def _on_resize(self, w: int, h: int) -> None:
        """Responsive: ajusta ancho en caracteres de los Entry y wraplengths para que no desborden en ventanas estrechas."""
        char_width = max(20, min(80, (w - 280) // 8))
        for entry in (self.entry_current, self.entry_hist, self.entry_out):
            try:
                entry.config(width=char_width)
            except (tk.TclError, AttributeError):
                pass
        
        # Ajustar wraplength del subheader dinámicamente
        if hasattr(self, 'subheader_label'):
            wraplength = max(400, w - 100)
            self.subheader_label.config(wraplength=wraplength)

    def _log(self, s: str):
        self.msg.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.msg.insert(tk.END, f"[{ts}] {s}\n")
        self.msg.see(tk.END)
        self.msg.config(state=tk.DISABLED)

    def _pick_current(self):
        p = filedialog.askopenfilename(title="Seleccionar Programas.xlsx", filetypes=[("Excel", "*.xlsx")], parent=self)
        if p:
            self.current_var.set(p)

    def _pick_hist(self):
        p = filedialog.askopenfilename(title="Seleccionar archivo histórico", filetypes=[("Excel", "*.xlsx")], parent=self)
        if p:
            self.hist_var.set(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(
            title="Guardar consolidado como...",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            parent=self,
        )
        if p:
            self.out_var.set(p)

    def _open_out(self):
        try:
            _open_in_excel(Path(self.out_var.get()))
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)

    def _merge(self):
        import pandas as pd  # Lazy import
        
        current_path = Path(self.current_var.get())
        hist_path = Path(self.hist_var.get())
        out_path = Path(self.out_var.get())

        if not current_path.exists():
            messagebox.showerror("Error", f"No existe: {current_path}", parent=self)
            return
        if not hist_path.exists():
            messagebox.showerror("Error", f"No existe: {hist_path}", parent=self)
            return

        try:
            from etl.exceptions_helpers import leer_excel_con_reintentos
            df_current = leer_excel_con_reintentos(current_path, sheet_name="Programas")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo leer el archivo actual: {exc}", parent=self)
            return
        try:
            # histórico de programas nuevos suele tener hoja ProgramasNuevos
            from etl.exceptions_helpers import leer_excel_con_reintentos
            try:
                df_hist = leer_excel_con_reintentos(hist_path, sheet_name="ProgramasNuevos")
            except Exception:
                df_hist = leer_excel_con_reintentos(hist_path)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo leer el histórico: {exc}", parent=self)
            return

        key = "CÓDIGO_SNIES_DEL_PROGRAMA"
        if key not in df_current.columns or key not in df_hist.columns:
            messagebox.showerror("Error", f"Ambos archivos deben tener la columna '{key}'.", parent=self)
            return

        df_current = df_current.copy()
        df_hist = df_hist.copy()
        df_current["ORIGEN_REGISTRO"] = "ACTUAL"
        df_hist["ORIGEN_REGISTRO"] = "HISTORICO"

        # Unificar columnas (union)
        all_cols = list(dict.fromkeys(list(df_current.columns) + list(df_hist.columns)))
        for c in all_cols:
            if c not in df_current.columns:
                df_current[c] = None
            if c not in df_hist.columns:
                df_hist[c] = None

        # Merge con regla de negocio:
        # 1) Si hay AJUSTE_MANUAL=True, eso gana sobre automático.
        # 2) Si no hay ajuste manual, gana ACTUAL sobre HISTORICO.
        # 3) Si hay FECHA_AJUSTE, se prefiere la más reciente.
        combined = pd.concat([df_hist[all_cols], df_current[all_cols]], ignore_index=True)

        def norm_codigo(v: object) -> str:
            s = str(v).strip() if v is not None else ""
            return s[:-2] if s.endswith(".0") else s

        combined["_CODIGO_NORM"] = combined[key].apply(norm_codigo)

        if "AJUSTE_MANUAL" not in combined.columns:
            combined["AJUSTE_MANUAL"] = False
        # Normalizar AJUSTE_MANUAL de forma segura:
        # - Excel puede traer bool, 0/1, o strings ("Sí"/"No", "true"/"false").
        def _to_bool(v: object) -> bool:
            if v is None:
                return False
            if isinstance(v, bool):
                return v
            try:
                # np.bool_ / ints / floats
                if isinstance(v, (int, float)):
                    return bool(int(v))
            except Exception:
                pass
            s = str(v).strip().lower()
            if s in ("1", "true", "t", "yes", "y", "si", "sí"):
                return True
            if s in ("0", "false", "f", "no", "n", ""):
                return False
            # fallback conservador: si no se entiende, asumir False
            return False

        combined["AJUSTE_MANUAL"] = combined["AJUSTE_MANUAL"].apply(_to_bool)

        if "FECHA_AJUSTE" not in combined.columns:
            combined["FECHA_AJUSTE"] = ""
        # Parse simple: si no parsea, queda NaT y se ordena al final
        combined["_FECHA_AJUSTE_TS"] = pd.to_datetime(combined["FECHA_AJUSTE"], errors="coerce")

        # Prioridades: manual primero, luego ACTUAL, luego fecha ajuste desc
        combined["_PRIO_MANUAL"] = combined["AJUSTE_MANUAL"].astype(int)
        combined["_PRIO_ORIGEN"] = (combined["ORIGEN_REGISTRO"].astype(str) == "ACTUAL").astype(int)

        combined = combined.sort_values(
            by=["_CODIGO_NORM", "_PRIO_MANUAL", "_PRIO_ORIGEN", "_FECHA_AJUSTE_TS"],
            ascending=[True, False, False, False],
        )
        combined["FUENTE_CONSOLIDADO"] = combined.apply(
            lambda r: "MANUAL" if bool(r.get("AJUSTE_MANUAL")) else str(r.get("ORIGEN_REGISTRO") or ""),
            axis=1,
        )
        combined = combined.drop_duplicates(subset=["_CODIGO_NORM"], keep="first")
        combined = combined.drop(columns=["_CODIGO_NORM", "_FECHA_AJUSTE_TS", "_PRIO_MANUAL", "_PRIO_ORIGEN"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pd.ExcelWriter(out_path, mode="w", engine="openpyxl") as writer:
                combined.to_excel(writer, sheet_name="Consolidado", index=False)
        except PermissionError:
            safe_messagebox_error("Error", explain_file_in_use(), parent=self)
            return

        self._log(f"Consolidado generado: {out_path} ({len(combined)} filas)")
        messagebox.showinfo("OK", f"Consolidado generado:\n{out_path}", parent=self)


class ImputationPage(ttk.Frame):
    """Página para imputar valores faltantes en ÁREA_DE_CONOCIMIENTO usando IA (KNN con embeddings)."""

    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        
        # Import lazy de módulos ETL (solo cuando se abre esta página)
        from etl.normalizacion import ARCHIVO_PROGRAMAS
        
        self.base_dir = ensure_base_dir(self)
        if not self.base_dir:
            if on_back:
                on_back()
            return

        self.file_path = ARCHIVO_PROGRAMAS
        self.is_running = False
        self.df_faltantes = None  # DataFrame con registros que tienen valores faltantes
        self.codigos_antes_imputacion = None  # Códigos SNIES de registros que tenían valores faltantes antes de la imputación

        frame = ttk.Frame(self, padding=14, style="Page.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(frame, style="Page.TFrame")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(header_frame, text="🤖 Revisión de Áreas (Imputación IA)", style="Header.TLabel").pack(side=tk.LEFT)
        if on_back:
            ttk.Button(header_frame, text="← Volver al menú", command=lambda: on_back() if on_back else None, style="Back.TButton").pack(side=tk.RIGHT)
        
        self.subheader_label = ttk.Label(
            frame, 
            text="Rellena valores faltantes en ÁREA_DE_CONOCIMIENTO usando KNN con embeddings semánticos. "
                 "El sistema encuentra los 5 programas más similares que tienen área asignada y asigna esa categoría.",
            style="SubHeader.TLabel"
        )
        self.subheader_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        # Información del archivo y estadísticas
        info_frame = ttk.Frame(frame, style="Page.TFrame")
        info_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(info_frame, text="Archivo a procesar:").pack(side=tk.LEFT, padx=(0, 8))
        self.file_label = ttk.Label(info_frame, text=str(self.file_path), style="Muted.TLabel", font=("Segoe UI", 9))
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Label de estadísticas
        self.stats_label = ttk.Label(
            info_frame, 
            text="", 
            style="Muted.TLabel", 
            font=("Segoe UI", 9, "bold"),
            foreground=EAFIT["azul_zafre"]
        )
        self.stats_label.pack(side=tk.RIGHT, padx=(8, 0))

        # Botones de acción
        btn_row = ttk.Frame(frame, style="Page.TFrame")
        btn_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        
        # Botones principales (izquierda)
        btn_left = ttk.Frame(btn_row, style="Page.TFrame")
        btn_left.pack(side=tk.LEFT)
        self.btn_imputar = ttk.Button(btn_left, text="🤖 Ejecutar Imputación IA", command=self._ejecutar_imputacion, style="Primary.TButton")
        self.btn_imputar.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_left, text="📂 Abrir en Excel", command=self._open_excel, style="Secondary.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_left, text="🔄 Recargar", command=self._recargar_info, style="Secondary.TButton").pack(side=tk.LEFT, padx=8)
        
        # NOTA: La imputación solo guarda en Programas.xlsx, NO actualiza el histórico
        # El histórico se actualiza automáticamente al finalizar el pipeline con programas nuevos

        # Tabla de registros con valores faltantes
        table_frame = ttk.Frame(frame, style="Page.TFrame")
        table_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        
        table_header = ttk.Frame(table_frame, style="Page.TFrame")
        table_header.pack(fill=tk.X, pady=(0, 4))
        self.table_title_label = ttk.Label(
            table_header, 
            text="📋 Registros con ÁREA_DE_CONOCIMIENTO faltante", 
            style="SubHeader.TLabel",
            font=("Segoe UI", 10, "bold")
        )
        self.table_title_label.pack(side=tk.LEFT)
        
        # Columnas a mostrar en la tabla
        self.table_columns = [
            "CÓDIGO_SNIES_DEL_PROGRAMA",
            "NOMBRE_INSTITUCIÓN",
            "NOMBRE_DEL_PROGRAMA",
            "NIVEL_DE_FORMACIÓN",
            "ÁREA_DE_CONOCIMIENTO",
        ]
        
        # Tabla de solo lectura (sin columnas editables)
        self.table = EditableTable(
            table_frame,
            columns=self.table_columns,
            height=15,
            editable_columns=set(),  # Sin columnas editables
            on_change=None,
        )
        self.table.pack(fill=tk.BOTH, expand=True)

        # Área de log (más pequeña ahora)
        log_frame = ttk.Frame(frame, style="Page.TFrame")
        log_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 0))
        ttk.Label(log_frame, text="📝 Log de ejecución:", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 4))
        self.msg = tk.Text(
            log_frame, 
            height=6, 
            wrap=tk.WORD, 
            state=tk.DISABLED, 
            font=("Consolas", 9), 
            bg=EAFIT["card_bg"], 
            fg=EAFIT["text"]
        )
        self.msg.pack(fill=tk.BOTH, expand=True)

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(4, weight=1)  # Tabla expandible
        frame.grid_rowconfigure(5, weight=0)  # Log fijo

        # Cargar información inicial
        self._recargar_info()

    def _on_resize(self, w: int, h: int) -> None:
        """Responsive: ajusta wraplengths y altura de tabla dinámicamente."""
        if hasattr(self, 'subheader_label'):
            wraplength = max(400, w - 100)
            self.subheader_label.config(wraplength=wraplength)
        if hasattr(self, 'file_label'):
            wraplength = max(300, w - 200)
            self.file_label.config(wraplength=wraplength)
        # Ajustar altura de la tabla: header + botones + log ~ 250px
        if hasattr(self, 'table'):
            table_pixels = max(150, h - 250)
            self.table.set_height_from_pixels(table_pixels)

    def _log(self, s: str):
        """Agrega un mensaje al área de log."""
        self.msg.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.msg.insert(tk.END, f"[{ts}] {s}\n")
        self.msg.see(tk.END)
        self.msg.config(state=tk.DISABLED)
        self.update_idletasks()

    def _recargar_info(self):
        """Recarga la información del archivo, muestra estadísticas y actualiza la tabla."""
        import pandas as pd
        from etl.exceptions_helpers import leer_excel_con_reintentos
        
        if not self.file_path.exists():
            self._log(f"⚠️ El archivo no existe: {self.file_path}")
            self.btn_imputar.config(state=tk.DISABLED)
            self._actualizar_tabla([])  # Limpiar tabla
            return
        
        try:
            df = leer_excel_con_reintentos(self.file_path, sheet_name="Programas")
            
            if "ÁREA_DE_CONOCIMIENTO" not in df.columns:
                self._log("⚠️ El archivo no tiene la columna 'ÁREA_DE_CONOCIMIENTO'")
                self.btn_imputar.config(state=tk.DISABLED)
                self._actualizar_tabla([])  # Limpiar tabla
                return
            
            # Usar la misma lógica que el módulo de imputación
            def _es_valor_faltante(valor: object) -> bool:
                if pd.isna(valor):
                    return True
                valor_str = str(valor).strip().lower()
                valores_faltantes = ["", "sin clasificar", "sin clasificacion", "n/a", "na", "none", "null"]
                return valor_str in valores_faltantes
            
            mask_faltantes = df["ÁREA_DE_CONOCIMIENTO"].apply(_es_valor_faltante)
            cantidad_faltantes = mask_faltantes.sum()
            total = len(df)
            
            # Filtrar registros con valores faltantes
            self.df_faltantes = df[mask_faltantes].copy()
            
            # Actualizar estadísticas en el label
            self.stats_label.config(
                text=f"Total: {total} | Con área: {total - cantidad_faltantes} | Faltantes: {cantidad_faltantes}"
            )
            
            self._log(f"📊 Estadísticas del archivo:")
            self._log(f"   Total de programas: {total}")
            self._log(f"   Con área asignada: {total - cantidad_faltantes}")
            self._log(f"   Sin área (faltantes): {cantidad_faltantes}")
            
            # Actualizar tabla con registros faltantes
            if cantidad_faltantes > 0:
                self._actualizar_tabla(self.df_faltantes)
                self._log(f"✓ Mostrando {cantidad_faltantes} registros con valores faltantes en la tabla.")
                self.btn_imputar.config(state=tk.NORMAL)
            else:
                self._actualizar_tabla([])
                self._log("✓ No hay valores faltantes. No se requiere imputación.")
                self.btn_imputar.config(state=tk.DISABLED)
                
        except Exception as exc:
            self._log(f"❌ Error al leer el archivo: {exc}")
            self.btn_imputar.config(state=tk.DISABLED)
            self._actualizar_tabla([])
    
    def _actualizar_tabla(self, df_faltantes, es_resultado_imputacion=False):
        """Actualiza la tabla con los registros que tienen valores faltantes o los resultados de imputación."""
        import pandas as pd
        
        # Limpiar tabla actual
        self.table.set_rows([])
        
        # Manejar diferentes tipos de entrada (DataFrame, lista vacía, None)
        if df_faltantes is None:
            titulo = "📋 Resultados de la imputación (0 registros)" if es_resultado_imputacion else "📋 Registros con ÁREA_DE_CONOCIMIENTO faltante (0 registros)"
            self.table_title_label.config(text=titulo)
            return
        
        # Si es una lista vacía
        if isinstance(df_faltantes, list) and len(df_faltantes) == 0:
            titulo = "📋 Resultados de la imputación (0 registros)" if es_resultado_imputacion else "📋 Registros con ÁREA_DE_CONOCIMIENTO faltante (0 registros)"
            self.table_title_label.config(text=titulo)
            return
        
        # Si es un DataFrame vacío
        if hasattr(df_faltantes, 'empty') and df_faltantes.empty:
            titulo = "📋 Resultados de la imputación (0 registros)" if es_resultado_imputacion else "📋 Registros con ÁREA_DE_CONOCIMIENTO faltante (0 registros)"
            self.table_title_label.config(text=titulo)
            return
        
        # Si es un DataFrame con datos
        if hasattr(df_faltantes, 'iterrows'):
            # Preparar datos para la tabla
            rows = []
            for _, row in df_faltantes.iterrows():
                row_dict = {}
                for col in self.table_columns:
                    valor = row.get(col, "")
                    # Convertir NaN a string vacío
                    if pd.isna(valor):
                        valor = ""
                    else:
                        valor = str(valor)
                    row_dict[col] = valor
                rows.append(row_dict)
            
            # Actualizar título de la tabla según el contexto
            if es_resultado_imputacion:
                self.table_title_label.config(
                    text=f"✅ Resultados de la imputación ({len(rows)} registros procesados)"
                )
            else:
                self.table_title_label.config(
                    text=f"📋 Registros con ÁREA_DE_CONOCIMIENTO faltante ({len(rows)} registros)"
                )
            
            # Establecer filas en la tabla
            self.table.set_rows(rows)
        else:
            # Si no es un DataFrame, intentar tratarlo como lista de diccionarios
            if isinstance(df_faltantes, list):
                self.table.set_rows(df_faltantes)
                if es_resultado_imputacion:
                    self.table_title_label.config(
                        text=f"✅ Resultados de la imputación ({len(df_faltantes)} registros procesados)"
                    )
                else:
                    self.table_title_label.config(
                        text=f"📋 Registros con ÁREA_DE_CONOCIMIENTO faltante ({len(df_faltantes)} registros)"
                    )
            else:
                titulo = "📋 Resultados de la imputación (0 registros)" if es_resultado_imputacion else "📋 Registros con ÁREA_DE_CONOCIMIENTO faltante (0 registros)"
                self.table_title_label.config(text=titulo)
    
    def _mostrar_resultados_imputacion(self, df_resultados, cantidad_imputados, faltantes_despues):
        """Muestra los resultados de la imputación en la tabla."""
        import pandas as pd
        
        # Preparar datos para la tabla
        rows = []
        for _, row in df_resultados.iterrows():
            row_dict = {}
            for col in self.table_columns:
                valor = row.get(col, "")
                # Convertir NaN a string vacío
                if pd.isna(valor):
                    valor = ""
                else:
                    valor = str(valor)
                row_dict[col] = valor
            rows.append(row_dict)
        
        # Actualizar título de la tabla
        self.table_title_label.config(
            text=f"✅ Resultados de la imputación ({len(rows)} registros procesados | {cantidad_imputados} imputados | {faltantes_despues} aún faltantes)"
        )
        
        # Establecer filas en la tabla
        self.table.set_rows(rows)
        
        # Log adicional
        self._log(f"📊 Mostrando {len(rows)} registros procesados en la tabla.")
        if cantidad_imputados > 0:
            self._log(f"   ✓ {cantidad_imputados} registros fueron imputados exitosamente.")
        if faltantes_despues > 0:
            self._log(f"   ⚠️ {faltantes_despues} registros aún tienen valores faltantes.")
    
    def _actualizar_estadisticas_despues(self, df_resultado):
        """Actualiza las estadísticas después de la imputación."""
        import pandas as pd
        
        def _es_valor_faltante(valor: object) -> bool:
            if pd.isna(valor):
                return True
            valor_str = str(valor).strip().lower()
            valores_faltantes = ["", "sin clasificar", "sin clasificacion", "n/a", "na", "none", "null"]
            return valor_str in valores_faltantes
        
        mask_faltantes = df_resultado["ÁREA_DE_CONOCIMIENTO"].apply(_es_valor_faltante)
        cantidad_faltantes = mask_faltantes.sum()
        total = len(df_resultado)
        
        # Actualizar estadísticas en el label
        self.stats_label.config(
            text=f"Total: {total} | Con área: {total - cantidad_faltantes} | Faltantes: {cantidad_faltantes}"
        )

    def _open_excel(self):
        """Abre el archivo Programas.xlsx en Excel."""
        try:
            _open_in_excel(self.file_path)
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo abrir el archivo:\n{exc}", parent=self)

    def _ejecutar_imputacion(self):
        """Ejecuta la imputación de áreas en un hilo separado."""
        if self.is_running:
            messagebox.showwarning("Atención", "La imputación ya está en ejecución.", parent=self)
            return
        
        if not self.file_path.exists():
            safe_messagebox_error("Error", f"El archivo no existe:\n{self.file_path}", parent=self)
            return
        
        # Confirmar antes de ejecutar
        if not _ask_yes_no(
            "Confirmar Imputación",
            f"¿Ejecutar imputación de ÁREA_DE_CONOCIMIENTO?\n\n"
            f"Esto puede tardar varios minutos dependiendo del número de valores faltantes.\n"
            f"El archivo {self.file_path.name} será modificado.",
            parent=self
        ):
            return
        
        self.is_running = True
        self.btn_imputar.config(state=tk.DISABLED)
        self._log("=" * 60)
        self._log("🚀 Iniciando imputación de ÁREA_DE_CONOCIMIENTO...")
        self._log("=" * 60)
        
        def ejecutar_en_hilo():
            try:
                from etl.imputacionAreas import ejecutar_imputacion_areas
                
                # Ejecutar imputación (modo archivo: lee y escribe directamente)
                self._log("📖 Leyendo archivo...")
                
                # Leer antes para contar faltantes
                import pandas as pd
                from etl.exceptions_helpers import leer_excel_con_reintentos
                df_antes = leer_excel_con_reintentos(self.file_path, sheet_name="Programas")
                
                def _es_valor_faltante(valor: object) -> bool:
                    if pd.isna(valor):
                        return True
                    valor_str = str(valor).strip().lower()
                    valores_faltantes = ["", "sin clasificar", "sin clasificacion", "n/a", "na", "none", "null"]
                    return valor_str in valores_faltantes
                
                faltantes_antes = df_antes["ÁREA_DE_CONOCIMIENTO"].apply(_es_valor_faltante).sum()
                
                # Guardar códigos SNIES de los registros que tenían valores faltantes ANTES de la imputación
                mask_faltantes_antes = df_antes["ÁREA_DE_CONOCIMIENTO"].apply(_es_valor_faltante)
                df_faltantes_antes = df_antes[mask_faltantes_antes].copy()
                
                # Normalizar códigos SNIES para comparación
                def _norm_codigo(v: object) -> str:
                    if v is None:
                        return ""
                    s = str(v).strip()
                    if s.endswith(".0"):
                        s = s[:-2]
                    return s
                
                codigos_antes = set(df_faltantes_antes["CÓDIGO_SNIES_DEL_PROGRAMA"].apply(_norm_codigo))
                
                self._log(f"📝 Registros con valores faltantes antes: {len(codigos_antes)}")
                
                # Ejecutar imputación
                self._log("🔄 Ejecutando imputación...")
                df_resultado = ejecutar_imputacion_areas(archivo=self.file_path)
                
                # Contar después
                faltantes_despues = df_resultado["ÁREA_DE_CONOCIMIENTO"].apply(_es_valor_faltante).sum()
                cantidad_imputados = faltantes_antes - faltantes_despues
                
                # Filtrar los registros que tenían valores faltantes ANTES (para mostrar resultados)
                df_resultado["_CODIGO_NORM"] = df_resultado["CÓDIGO_SNIES_DEL_PROGRAMA"].apply(_norm_codigo)
                df_resultados_imputacion = df_resultado[df_resultado["_CODIGO_NORM"].isin(codigos_antes)].copy()
                df_resultados_imputacion = df_resultados_imputacion.drop(columns=["_CODIGO_NORM"])
                
                # Filtrar registros que aún tienen valores faltantes (si los hay)
                mask_faltantes_despues = df_resultado["ÁREA_DE_CONOCIMIENTO"].apply(_es_valor_faltante)
                df_faltantes_despues = df_resultado[mask_faltantes_despues].copy()
                
                self._log("=" * 60)
                self._log("✅ Imputación completada exitosamente!")
                self._log(f"   Valores imputados: {cantidad_imputados}")
                self._log(f"   Valores aún faltantes: {faltantes_despues}")
                self._log(f"   Archivo actualizado: {self.file_path.name}")
                self._log("=" * 60)
                
                # Actualizar tabla con los resultados de la imputación
                def actualizar_ui():
                    # Mostrar los registros que fueron procesados (con sus nuevos valores imputados)
                    self._mostrar_resultados_imputacion(df_resultados_imputacion, cantidad_imputados, faltantes_despues)
                    
                    # Actualizar estadísticas
                    self._actualizar_estadisticas_despues(df_resultado)
                    
                    # Mostrar mensaje de éxito
                    messagebox.showinfo(
                        "Imputación Completada",
                        f"La imputación se completó exitosamente.\n\n"
                        f"Valores imputados: {cantidad_imputados}\n"
                        f"Valores aún faltantes: {faltantes_despues}\n"
                        f"Archivo actualizado: {self.file_path.name}\n\n"
                        f"Los resultados se muestran en la tabla.\n\n"
                        f"💡 Puedes usar el botón 'Actualizar Histórico' para agregar los programas nuevos al histórico.",
                        parent=self
                    )
                
                self.after(0, actualizar_ui)
                
            except Exception as exc:
                error_msg = str(exc)
                self._log("=" * 60)
                self._log(f"❌ Error durante la imputación: {error_msg}")
                self._log("=" * 60)
                self.after(0, lambda: safe_messagebox_error("Error", f"Error durante la imputación:\n{error_msg}", parent=self))
            finally:
                self.after(0, lambda: self._finalizar_imputacion())
        
        # Ejecutar en hilo separado para no bloquear la UI
        thread = threading.Thread(target=ejecutar_en_hilo, daemon=True)
        thread.start()

    def _finalizar_imputacion(self):
        """Restaura el estado de la UI después de la imputación."""
        self.is_running = False
        self.btn_imputar.config(state=tk.NORMAL)
        # No recargar automáticamente aquí porque los resultados ya se muestran en actualizar_ui()
        # Si el usuario quiere ver el estado actualizado, puede usar el botón "Recargar"
    
        # NOTA: La función _actualizar_historico fue eliminada porque la imputación
        # solo debe guardar en Programas.xlsx, NO en el histórico.
        # El histórico se actualiza automáticamente al finalizar el pipeline.


class ConfiguracionDialog(tk.Toplevel):
    """
    Diálogo modal de configuración del sistema.
    Permite cambiar AÑO_FIN_DATOS, ver estado de archivos en ref/backup/ y actualizar SMLMV_POR_ANO.
    Los cambios se guardan en config.json.
    """

    _ARCHIVOS_REQUERIDOS: dict[str, str] = {
        "Matrícula": "matriculados_{año}.xlsx",
        "Inscritos": "inscritos_{año}.xlsx",
        "Primer curso": "primer_curso_{año}.xlsx",
        "Graduados": "graduados_{año}.xlsx",
    }

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("⚙️ Configuración del Sistema")
        self.resizable(False, False)
        self.configure(bg=EAFIT["bg"])

        self.update_idletasks()
        w, h = 620, 600
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        self._cfg = self._cargar_config()
        try:
            from etl.config import AÑO_FIN_DATOS as _afd

            _def_año = int(_afd)
        except Exception:
            _def_año = 2024
        self._año_var = tk.IntVar(value=int(self._cfg.get("AÑO_FIN_DATOS", _def_año)))
        self._smlmv_vars: dict[int, tk.StringVar] = {}
        self._file_labels: dict[str, tuple[ttk.Label, ttk.Label]] = {}

        self._build_ui()
        self._año_var.trace_add("write", lambda *_: self.after_idle(self._refresh_archivos))
        self._refresh_archivos()

    @staticmethod
    def _cargar_config() -> dict:
        try:
            from etl.config import CONFIG_PATH

            if CONFIG_PATH.exists():
                return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @staticmethod
    def _guardar_config(cfg: dict) -> None:
        from etl.config import CONFIG_PATH

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _get_ref_backup() -> Path:
        from etl.config import REF_DIR

        return REF_DIR / "backup"

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=24, style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Configuración del Sistema", style="Header.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            main,
            text="Los cambios se guardan en config.json y se aplican al guardar (año y SMLMV en esta sesión).",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(0, 18))

        sec1 = ttk.Frame(main, padding=18, style="Card.TFrame")
        sec1.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(sec1, text="📅  Año de datos activo", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 10))

        año_row = ttk.Frame(sec1, style="Card.TFrame")
        año_row.pack(fill=tk.X)

        ttk.Label(año_row, text="AÑO_FIN_DATOS:", style="Muted.TLabel", font=("Segoe UI", 10)).pack(side=tk.LEFT)

        spin = ttk.Spinbox(
            año_row,
            from_=2019,
            to=2035,
            textvariable=self._año_var,
            width=8,
            font=("Segoe UI", 11, "bold"),
        )
        spin.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(
            año_row,
            text="  ← cambia el año cuando lleguen los archivos nuevos",
            style="Muted.TLabel",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(8, 0))

        sec2 = ttk.Frame(main, padding=18, style="Card.TFrame")
        sec2.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(sec2, text="📁  Archivos disponibles en ref/backup/", style="SectionTitle.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        grid = ttk.Frame(sec2, style="Card.TFrame")
        grid.pack(fill=tk.X)

        for i, (tipo, patron) in enumerate(self._ARCHIVOS_REQUERIDOS.items()):
            row = i // 2
            col = (i % 2) * 3

            ttk.Label(grid, text=tipo + ":", style="Muted.TLabel", width=15, anchor="w").grid(
                row=row, column=col, sticky="w", padx=(0, 4), pady=3
            )

            lbl_nombre = ttk.Label(grid, text="—", style="Muted.TLabel", font=("Segoe UI", 9), width=28, anchor="w")
            lbl_nombre.grid(row=row, column=col + 1, sticky="w", pady=3)

            lbl_estado = ttk.Label(
                grid, text="", style="Muted.TLabel", font=("Segoe UI", 10, "bold"), width=4, anchor="center"
            )
            lbl_estado.grid(row=row, column=col + 2, sticky="w", padx=(0, 16), pady=3)

            self._file_labels[tipo] = (lbl_nombre, lbl_estado)

        self._lbl_resumen = ttk.Label(sec2, text="", style="Muted.TLabel", font=("Segoe UI", 9, "italic"))
        self._lbl_resumen.pack(anchor="w", pady=(8, 0))

        sec3 = ttk.Frame(main, padding=18, style="Card.TFrame")
        sec3.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(sec3, text="💰  SMLMV por año (para cálculo de salarios)", style="SectionTitle.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        smlmv_grid = ttk.Frame(sec3, style="Card.TFrame")
        smlmv_grid.pack(fill=tk.X)

        try:
            from etl.config import SMLMV_POR_ANO

            smlmv_base = dict(SMLMV_POR_ANO)
        except Exception:
            smlmv_base = {2023: 1_160_000, 2024: 1_300_000, 2025: 1_423_500}

        cfg_smlmv = self._cfg.get("SMLMV_POR_ANO", {})
        if isinstance(cfg_smlmv, dict):
            for k, v in cfg_smlmv.items():
                try:
                    smlmv_base[int(k)] = int(v)
                except (TypeError, ValueError):
                    pass

        años_smlmv = sorted(smlmv_base.keys())
        for i, año in enumerate(años_smlmv):
            row = i // 3
            col = (i % 3) * 3
            var = tk.StringVar(value=f"{smlmv_base[año]:,}".replace(",", "."))
            self._smlmv_vars[año] = var

            ttk.Label(smlmv_grid, text=f"{año}:", style="Muted.TLabel", width=6, anchor="e").grid(
                row=row, column=col, padx=(8, 2), pady=2, sticky="e"
            )
            ttk.Entry(smlmv_grid, textvariable=var, width=12, font=("Segoe UI", 9)).grid(
                row=row, column=col + 1, padx=(0, 8), pady=2, sticky="w"
            )

        ttk.Label(
            sec3,
            text="Formato: número sin puntos ni comas (ej. 1423500) o con puntos como miles (1.423.500).",
            style="Light.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        btn_row = ttk.Frame(main, style="App.TFrame")
        btn_row.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(btn_row, text="Cancelar", command=self.destroy, style="Secondary.TButton").pack(
            side=tk.RIGHT, padx=(8, 0)
        )

        ttk.Button(btn_row, text="✅  Guardar y aplicar", command=self._guardar, style="Primary.TButton").pack(
            side=tk.RIGHT
        )

    def _refresh_archivos(self) -> None:
        try:
            año = int(self._año_var.get())
        except (ValueError, tk.TclError):
            return

        backup = self._get_ref_backup()
        presentes = 0
        total = len(self._ARCHIVOS_REQUERIDOS)

        for tipo, patron in self._ARCHIVOS_REQUERIDOS.items():
            nombre = patron.format(año=año)
            candidatos = [
                backup / nombre,
                backup / "matriculas" / nombre,
                backup / "matriculas primer curso" / nombre,
                backup / "inscritos" / nombre,
                backup / "graduados" / nombre,
            ]
            existe = any(c.exists() for c in candidatos)
            if existe:
                presentes += 1

            lbl_nombre, lbl_estado = self._file_labels[tipo]
            lbl_nombre.configure(text=nombre)
            lbl_estado.configure(
                text="✅" if existe else "❌",
                foreground=EAFIT["success"] if existe else EAFIT["danger"],
            )

        if presentes == total:
            resumen = f"✅ Todos los archivos de {año} están disponibles — puedes aplicar el cambio."
            color = EAFIT["success"]
        elif presentes == 0:
            resumen = f"❌ No se encontró ningún archivo de {año} en ref/backup/."
            color = EAFIT["danger"]
        else:
            resumen = (
                f"⚠️ {presentes}/{total} archivos de {año} disponibles. "
                "El pipeline usará NaN donde falten datos."
            )
            color = EAFIT["warning"]

        self._lbl_resumen.configure(text=resumen, foreground=color)

    def _guardar(self) -> None:
        try:
            año_nuevo = int(self._año_var.get())
            if año_nuevo < 2019 or año_nuevo > 2035:
                raise ValueError("Año fuera de rango")
        except (ValueError, tk.TclError):
            messagebox.showerror("Error", "El año debe ser un número entre 2019 y 2035.", parent=self)
            return

        smlmv_nuevo: dict[int, int] = {}
        for año_s, var in self._smlmv_vars.items():
            try:
                raw = str(var.get()).strip()
                valor = int(raw.replace(".", "").replace(",", ""))
                if valor > 0:
                    smlmv_nuevo[año_s] = valor
            except ValueError:
                messagebox.showerror("Error", f"SMLMV {año_s}: valor inválido '{var.get()}'.", parent=self)
                return

        cfg = self._cargar_config()
        cfg["AÑO_FIN_DATOS"] = año_nuevo
        if smlmv_nuevo:
            cfg["SMLMV_POR_ANO"] = {str(k): v for k, v in sorted(smlmv_nuevo.items())}

        try:
            self._guardar_config(cfg)
            try:
                from etl.config import reload_year_and_smlmv_from_config_file

                reload_year_and_smlmv_from_config_file()
            except Exception:
                pass
            messagebox.showinfo(
                "Configuración guardada",
                f"AÑO_FIN_DATOS = {año_nuevo}\n\n"
                "Los valores de año y SMLMV quedaron activos en esta sesión. "
                "Si algo no se refleja, reinicia la aplicación.",
                parent=self,
            )
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e), parent=self)


class GestionProgramasDialog(tk.Toplevel):
    """
    Diálogo para gestionar programas candidatos en el análisis de valorización.
    Edita ref/backup/programas_para_valorizacion.xlsx directamente.
    """

    _COL_CATEGORIA = "Categoría"
    _COL_NIVEL = "Nivel"
    _COL_PROGRAMA = "Programas en proceso Calidad académica "
    _COL_ESTUDIO = "Tiene estudio de mercado "
    _COL_REGION = "Región"
    _REGIONES = ["Antioquia", "Bogota", "Eje Cafetero", "Virtual"]
    _SHEET = "Hoja1"

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("➕ Añadir programas")
        self.resizable(True, True)
        self.configure(bg=EAFIT["bg"])

        w, h = 780, 800
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(720, 720)

        self._ruta = self._encontrar_archivo()
        self._df = self._cargar()
        self._modificado = False

        self._build_ui()
        self._refrescar_tabla()

    @staticmethod
    def _encontrar_archivo() -> Path:
        from etl.config import REF_DIR

        candidatos = [
            REF_DIR / "backup" / "programas_para_valorizacion.xlsx",
            REF_DIR / "programas_para_valorizacion.xlsx",
        ]
        for p in candidatos:
            if p.exists():
                return p
        return candidatos[0]

    def _cargar(self) -> pd.DataFrame:
        cols_df = [self._COL_CATEGORIA, self._COL_NIVEL, self._COL_PROGRAMA, self._COL_ESTUDIO]
        if not self._ruta.exists():
            return pd.DataFrame(columns=cols_df)

        try:
            df = pd.read_excel(self._ruta, header=1, usecols=range(5))
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}", parent=self)
            return pd.DataFrame(columns=cols_df)

        rename: dict[str, str] = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if "categor" in cl:
                rename[c] = self._COL_CATEGORIA
            elif cl == "nivel" or "nivel" in cl:
                rename[c] = self._COL_NIVEL
            elif "proceso" in cl or "calidad" in cl or (
                "programa" in cl and "eafit" not in cl
            ):
                rename[c] = self._COL_PROGRAMA
            elif "estudio" in cl or "mercado" in cl:
                rename[c] = self._COL_ESTUDIO
            elif "regi" in cl:
                rename[c] = self._COL_REGION
        df = df.rename(columns=rename)

        for col in cols_df:
            if col not in df.columns:
                df[col] = "" if col != self._COL_ESTUDIO else "No"

        df[self._COL_PROGRAMA] = df[self._COL_PROGRAMA].astype(str).str.strip()
        df = df[df[self._COL_PROGRAMA].astype(str).str.len() > 0]

        # Expandir categorías compuestas si hay categorías válidas disponibles
        try:
            from etl.valorizacion_pipeline import (
                _categorias_validas_desde_estudio,
                _expandir_categorias_compuestas,
            )

            _cats_validas_gui = _categorias_validas_desde_estudio()
            if _cats_validas_gui:
                df = _expandir_categorias_compuestas(
                    df, _cats_validas_gui, col_cat=self._COL_CATEGORIA
                )
        except Exception:
            pass  # Si falla, continuar sin expandir (no bloquear la GUI)

        df = df.drop_duplicates(
            subset=[self._COL_PROGRAMA, self._COL_CATEGORIA], keep="first"
        ).reset_index(drop=True)
        return df[cols_df]

    def _guardar(self) -> bool:
        try:
            self._ruta.parent.mkdir(parents=True, exist_ok=True)
            filas: list[dict[str, str]] = []
            for _, row in self._df.iterrows():
                for reg in self._REGIONES:
                    filas.append(
                        {
                            self._COL_CATEGORIA: str(row.get(self._COL_CATEGORIA, "") or "").strip(),
                            self._COL_NIVEL: str(row.get(self._COL_NIVEL, "") or "").strip(),
                            self._COL_PROGRAMA: str(row[self._COL_PROGRAMA]).strip(),
                            self._COL_ESTUDIO: str(row.get(self._COL_ESTUDIO, "No") or "No").strip(),
                            self._COL_REGION: reg,
                        }
                    )
            df_save = pd.DataFrame(filas)
            hdr = [
                self._COL_CATEGORIA,
                self._COL_NIVEL,
                self._COL_PROGRAMA,
                self._COL_ESTUDIO,
                self._COL_REGION,
            ]
            df_save = df_save[hdr]

            with pd.ExcelWriter(self._ruta, engine="openpyxl") as writer:
                pd.DataFrame([[None] * len(hdr)]).to_excel(
                    writer, sheet_name=self._SHEET, index=False, header=False
                )
                df_save.to_excel(writer, sheet_name=self._SHEET, index=False, startrow=1)

            self._modificado = False
            return True
        except PermissionError:
            messagebox.showerror(
                "Archivo abierto",
                "No se pudo guardar porque el archivo está abierto en Excel.\n"
                "Ciérralo y vuelve a intentar.",
                parent=self,
            )
            return False
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e), parent=self)
            return False

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=20, style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Añadir programas a valorizar", style="Header.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            main,
            text="Lista de candidatos que EAFIT evalúa abrir. Se usan en la Fase 7 (Valorización).",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        # ── Pie fijo: botones Guardar/Cancelar ──────────────────────
        btn_row = ttk.Frame(main, style="App.TFrame")
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))

        self._lbl_estado = ttk.Label(btn_row, text="", style="Muted.TLabel", font=("Segoe UI", 9))
        self._lbl_estado.pack(side=tk.LEFT)

        ttk.Button(
            btn_row,
            text="Cancelar",
            command=self._confirmar_cierre,
            style="Secondary.TButton",
        ).pack(side=tk.RIGHT, padx=(8, 0))

        ttk.Button(
            btn_row,
            text="💾  Guardar cambios",
            command=self._guardar_y_cerrar,
            style="Primary.TButton",
        ).pack(side=tk.RIGHT)

        # ── Formulario fijo encima del pie (no lo comprime la tabla) ─
        add_frame = ttk.Frame(main, padding=(12, 10, 12, 10), style="Card.TFrame")
        add_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 10))

        ttk.Label(add_frame, text="Añadir programa candidato", style="SectionTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        campos = ttk.Frame(add_frame, style="Card.TFrame")
        campos.pack(fill=tk.X)

        ttk.Label(campos, text="Nombre del programa:", style="Body.TLabel", width=22).grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self._entry_nombre = ttk.Entry(campos, font=("Segoe UI", 10), width=46)
        self._entry_nombre.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(campos, text="Categoría de mercado:", style="Body.TLabel", width=22).grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self._entry_categoria = ttk.Entry(campos, font=("Segoe UI", 10), width=46)
        self._entry_categoria.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(campos, text="Categoría 2 (opcional):", style="Body.TLabel", width=22).grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self._entry_categoria2 = ttk.Entry(campos, font=("Segoe UI", 10), width=46)
        self._entry_categoria2.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(
            campos,
            text="Si se llena, se crean dos registros independientes.",
            style="Muted.TLabel",
            font=("Segoe UI", 8),
        ).grid(row=3, column=1, sticky="w", pady=(0, 4))

        ttk.Label(campos, text="Nivel:", style="Body.TLabel", width=22).grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self._nivel_var = tk.StringVar(value="Maestría")
        nivel_cb = ttk.Combobox(
            campos,
            textvariable=self._nivel_var,
            values=["Especialización", "Maestría", "Doctorado", "Universitario"],
            state="readonly",
            width=20,
        )
        nivel_cb.grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(campos, text="¿Tiene estudio?:", style="Body.TLabel", width=22).grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=4
        )
        estudio_frame = ttk.Frame(campos, style="Card.TFrame")
        estudio_frame.grid(row=5, column=1, sticky="w", pady=4)
        self._estudio_var = tk.StringVar(value="No")
        ttk.Radiobutton(estudio_frame, text="Sí", variable=self._estudio_var, value="Si").pack(
            side=tk.LEFT
        )
        ttk.Radiobutton(estudio_frame, text="No", variable=self._estudio_var, value="No").pack(
            side=tk.LEFT, padx=(12, 0)
        )
        campos.columnconfigure(1, weight=1)

        add_btn_row = ttk.Frame(add_frame, style="Card.TFrame")
        add_btn_row.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(
            add_btn_row,
            text="➕ Añadir a la lista",
            command=self._añadir_programa,
            style="Primary.TButton",
        ).pack(side=tk.LEFT)

        ttk.Button(
            add_btn_row,
            text="📂 Importar desde Excel...",
            command=self._importar_excel,
            style="Secondary.TButton",
        ).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(
            add_btn_row,
            text="El Excel de importación debe tener una columna con los nombres.",
            style="Muted.TLabel",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=(14, 0))

        # ── Tabla: solo el espacio central restante ─────────────────
        tabla_frame = ttk.Frame(main, style="Card.TFrame", padding=12)
        tabla_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Label(tabla_frame, text="Programas actuales", style="SectionTitle.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        tree_frame = ttk.Frame(tabla_frame, style="Card.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("programa", "categoria", "nivel", "estudio"),
            show="headings",
            height=6,
            selectmode="browse",
        )
        self._tree.heading("programa", text="Nombre del Programa")
        self._tree.heading("categoria", text="Categoría")
        self._tree.heading("nivel", text="Nivel")
        self._tree.heading("estudio", text="¿Tiene estudio?")
        self._tree.column("programa", width=300, anchor="w")
        self._tree.column("categoria", width=180, anchor="w")
        self._tree.column("nivel", width=110, anchor="w")
        self._tree.column("estudio", width=90, anchor="center")

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            tabla_frame,
            text="🗑 Eliminar seleccionado",
            command=self._eliminar_seleccionado,
            style="Secondary.TButton",
        ).pack(anchor="e", pady=(8, 0))

        self.protocol("WM_DELETE_WINDOW", self._confirmar_cierre)

    def _refrescar_tabla(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for _, row in self._df.iterrows():
            self._tree.insert(
                "",
                tk.END,
                values=(
                    row[self._COL_PROGRAMA],
                    row.get(self._COL_CATEGORIA, ""),
                    row.get(self._COL_NIVEL, ""),
                    row.get(self._COL_ESTUDIO, ""),
                ),
            )
        n = len(self._df)
        estado = f"{n} programa{'s' if n != 1 else ''} en la lista"
        if self._modificado:
            estado += "  ·  ⚠️ Cambios sin guardar"
        self._lbl_estado.configure(text=estado)

    def _añadir_programa(self) -> None:
        nombre = self._entry_nombre.get().strip()
        if not nombre:
            messagebox.showwarning(
                "Campo vacío", "Escribe el nombre del programa.", parent=self
            )
            return

        cat1 = self._entry_categoria.get().strip()
        cat2 = self._entry_categoria2.get().strip()

        if not cat1:
            messagebox.showwarning(
                "Campo vacío",
                "Indica al menos una categoría de mercado "
                "(ej. ANALITICA DE DATOS).",
                parent=self,
            )
            return

        categorias = [cat1]
        if cat2:
            categorias.append(cat2)

        existing = set(
            zip(
                self._df[self._COL_PROGRAMA].astype(str).str.strip().str.lower(),
                self._df[self._COL_CATEGORIA].astype(str).str.strip().str.lower(),
            )
        )

        nuevas_filas: list[dict] = []
        ya_existen: list[str] = []
        for cat in categorias:
            key = (nombre.lower(), cat.lower())
            if key in existing:
                ya_existen.append(cat)
            else:
                nuevas_filas.append(
                    {
                        self._COL_CATEGORIA: cat,
                        self._COL_NIVEL: self._nivel_var.get().strip(),
                        self._COL_PROGRAMA: nombre,
                        self._COL_ESTUDIO: self._estudio_var.get(),
                    }
                )

        if ya_existen:
            messagebox.showinfo(
                "Duplicado",
                f'"{nombre}" ya existe con la categoría: '
                f'{", ".join(ya_existen)}.',
                parent=self,
            )

        if nuevas_filas:
            self._df = pd.concat(
                [self._df, pd.DataFrame(nuevas_filas)], ignore_index=True
            )
            self._modificado = True
            self._entry_nombre.delete(0, tk.END)
            self._entry_categoria.delete(0, tk.END)
            self._entry_categoria2.delete(0, tk.END)
            self._refrescar_tabla()

    def _eliminar_seleccionado(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        nombre = self._tree.item(sel[0], "values")[0]
        if not messagebox.askyesno(
            "Confirmar", f'¿Eliminar "{nombre}" de la lista?', parent=self
        ):
            return
        mask = self._df[self._COL_PROGRAMA].astype(str) != nombre
        self._df = self._df.loc[mask].reset_index(drop=True)
        self._modificado = True
        self._refrescar_tabla()

    def _importar_excel(self) -> None:
        self._cats_validas_importar: set[str] = set()
        try:
            from etl.valorizacion_pipeline import (
                _categorias_validas_desde_estudio,
                _norm_cat,
            )

            _cats_list = _categorias_validas_desde_estudio()
            self._cats_validas_importar = {_norm_cat(c) for c in _cats_list}
            self._cats_canon_importar = {_norm_cat(c): c for c in _cats_list}
        except Exception:
            self._cats_canon_importar = {}

        ruta = filedialog.askopenfilename(
            title="Seleccionar Excel con programas candidatos",
            filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")],
            parent=self,
        )
        if not ruta:
            return
        try:
            df_imp = pd.read_excel(ruta)
        except Exception as e:
            messagebox.showerror("Error al leer", str(e), parent=self)
            return

        col_prog = None
        for c in df_imp.columns:
            cl = str(c).strip().lower()
            if any(k in cl for k in ("programa", "nombre", "proceso", "calidad")):
                col_prog = c
                break
        if col_prog is None:
            col_prog = df_imp.columns[0]

        col_cat = None
        for c in df_imp.columns:
            cl = str(c).strip().lower()
            if "categor" in cl:
                col_cat = c
                break

        col_cat2 = None
        for c in df_imp.columns:
            cl = str(c).strip().lower()
            if "categor" in cl and cl != str(col_cat).strip().lower():
                col_cat2 = c
                break

        col_niv = None
        for c in df_imp.columns:
            cl = str(c).strip().lower()
            if cl == "nivel" or cl.startswith("nivel"):
                col_niv = c
                break

        col_est = None
        for c in df_imp.columns:
            cl = str(c).strip().lower()
            if "estudio" in cl or "mercado" in cl:
                col_est = c
                break

        nombres_actuales = set(
            zip(
                self._df[self._COL_PROGRAMA].astype(str).str.strip().str.lower(),
                self._df[self._COL_CATEGORIA].astype(str).str.strip().str.lower(),
            )
        )
        nuevos: list[dict[str, str]] = []
        duplicados = 0
        for _, row in df_imp.iterrows():
            nombre = str(row[col_prog]).strip()
            if not nombre or nombre.lower() == "nan":
                continue
            estudio = str(row[col_est]).strip() if col_est else "No"
            if estudio.lower() not in ("si", "sí", "yes", "true", "1"):
                estudio = "No"
            else:
                estudio = "Si"
            cat_raw = str(row[col_cat]).strip() if col_cat and pd.notna(row.get(col_cat)) else ""
            niv = (
                str(row[col_niv]).strip()
                if col_niv and pd.notna(row.get(col_niv))
                else "Maestría"
            )

            # Split de categoría compuesta al importar
            if "-" in cat_raw and getattr(self, "_cats_validas_importar", None):
                import re

                from etl.valorizacion_pipeline import _norm_cat

                partes_imp = [p.strip() for p in re.split(r"\s*-\s*", cat_raw)]
                partes_imp = [p for p in partes_imp if p]
                canon = getattr(self, "_cats_canon_importar", {})
                validas_imp = [
                    canon[_norm_cat(p)]
                    for p in partes_imp
                    if _norm_cat(p) in self._cats_validas_importar
                ]
                cats_a_crear = validas_imp if validas_imp else [cat_raw]
            else:
                cats_a_crear = [cat_raw]

            if col_cat2 and pd.notna(row.get(col_cat2)):
                cat2_val = str(row[col_cat2]).strip()
                if cat2_val and cat2_val.lower() != "nan":
                    cats_a_crear.append(cat2_val)

            for cat_item in cats_a_crear:
                key = (nombre.lower(), cat_item.lower())
                if key not in nombres_actuales:
                    nuevos.append(
                        {
                            self._COL_CATEGORIA: cat_item,
                            self._COL_NIVEL: niv,
                            self._COL_PROGRAMA: nombre,
                            self._COL_ESTUDIO: estudio,
                        }
                    )
                    nombres_actuales.add(key)
                else:
                    duplicados += 1

        if not nuevos:
            msg = "No se encontraron programas nuevos."
            if duplicados:
                msg += f"\n{duplicados} ya estaban en la lista."
            messagebox.showinfo("Importación", msg, parent=self)
            return

        self._df = pd.concat([self._df, pd.DataFrame(nuevos)], ignore_index=True)
        self._modificado = True
        self._refrescar_tabla()

        msg = f"✅ {len(nuevos)} programa(s) importado(s)."
        if duplicados:
            msg += f"\n{duplicados} omitido(s) por duplicado."
        messagebox.showinfo("Importación completada", msg, parent=self)

    def _guardar_y_cerrar(self) -> None:
        if self._guardar():
            messagebox.showinfo(
                "Guardado",
                f"Lista guardada con {len(self._df)} programas.\n"
                "Los cambios se aplicarán en la próxima ejecución de la Fase 7.",
                parent=self,
            )
            self.destroy()

    def _confirmar_cierre(self) -> None:
        if self._modificado:
            if messagebox.askyesno(
                "Cambios sin guardar",
                "Hay cambios sin guardar. ¿Cerrar de todas formas?",
                parent=self,
            ):
                self.destroy()
        else:
            self.destroy()


class MainMenuGUI:
    """Menú principal del sistema."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root._main_menu_gui = self
        self.root.title("Clasificación de Programas SNIES - EAFIT")
        self.root.geometry("1200x720")
        self.root.minsize(900, 600)  # Tamaño mínimo más generoso para mejor visualización
        self.root.resizable(True, True)
        apply_modern_style(self.root)

        self.outer = ttk.Frame(root, padding=18, style="App.TFrame")
        self.outer.pack(fill=tk.BOTH, expand=True)
        
        # Contenedor para páginas (se muestra/oculta según la acción)
        self.content_container = ttk.Frame(self.outer, style="App.TFrame")
        # Inicialmente oculto, se mostrará cuando se abra una página
        # self.content_container.pack(fill=tk.BOTH, expand=True)  # No empaquetar inicialmente
        
        # Frame del menú principal (inicialmente visible)
        # Estructura: contenido scrollable + footer fijo siempre visible
        self.menu_frame = ttk.Frame(self.outer, style="App.TFrame")
        self.menu_frame.pack(fill=tk.BOTH, expand=True)
        
        # Contenedor para el contenido scrollable (sin el footer)
        self.menu_content_container = ttk.Frame(self.menu_frame, style="App.TFrame")
        self.menu_content_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas y scrollbar para contenido scrollable (dentro del contenedor, no en menu_frame directamente)
        self.menu_canvas = tk.Canvas(self.menu_content_container, highlightthickness=0, bg=EAFIT["bg"])
        self.menu_scrollbar = ttk.Scrollbar(self.menu_content_container, orient="vertical", command=self.menu_canvas.yview)
        
        # Frame contenedor para centrar el contenido y limitar ancho máximo
        self.menu_content_wrapper = ttk.Frame(self.menu_canvas, style="App.TFrame")
        
        # Frame para el contenido scrollable (todo excepto el footer)
        # Padding más generoso para mejor espaciado visual
        self.menu_content = ttk.Frame(self.menu_content_wrapper, padding=24, style="App.TFrame")
        self.menu_content.pack(fill=tk.BOTH, expand=True)
        
        # Configurar scroll y ajuste de ancho del canvas
        def _configure_canvas(event=None):
            # Ajustar ancho del contenido al canvas completo
            canvas_width = self.menu_canvas.winfo_width()
            if canvas_width > 1:
                # Usar TODO el ancho disponible del canvas (sin restricciones)
                # El scrollbar se manejará automáticamente
                self.menu_canvas.itemconfig(self.menu_content_window, width=canvas_width)
            # Actualizar scrollregion y visibilidad del scrollbar
            # El scrollregion solo debe considerar el contenido, no el footer
            self.menu_canvas.configure(scrollregion=self.menu_canvas.bbox("all"))
            _update_scrollbar_visibility()
            # Forzar actualización responsive después de cambiar el ancho
            self.root.after_idle(self._update_responsive)
        
        # Función para mostrar/ocultar scrollbar según necesidad
        def _update_scrollbar_visibility():
            try:
                bbox = self.menu_canvas.bbox("all")
                if bbox:
                    canvas_height = self.menu_canvas.winfo_height()
                    content_height = bbox[3] - bbox[1]
                    if content_height > canvas_height and canvas_height > 0:
                        # Mostrar scrollbar solo si el contenido es más alto que el canvas
                        if not self.menu_scrollbar.winfo_ismapped():
                            self.menu_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    else:
                        # Ocultar scrollbar si no es necesario
                        if self.menu_scrollbar.winfo_ismapped():
                            self.menu_scrollbar.pack_forget()
            except Exception:
                pass
        
        self.menu_content_wrapper.bind("<Configure>", _configure_canvas)
        
        self.menu_content_window = self.menu_canvas.create_window((0, 0), window=self.menu_content_wrapper, anchor="nw")
        self.menu_canvas.configure(yscrollcommand=self.menu_scrollbar.set)
        
        # Bind para ajustar ancho cuando cambia el tamaño del canvas
        self.menu_canvas.bind("<Configure>", _configure_canvas)
        # También actualizar cuando cambie el tamaño de la ventana principal
        self.root.bind("<Configure>", lambda e: self.root.after_idle(_configure_canvas) if e.widget == self.root else None)
        
        # Empacar canvas y scrollbar dentro del contenedor de contenido
        # El scrollbar se mostrará/ocultará automáticamente según necesidad en _configure_canvas
        self.menu_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # El scrollbar se empaqueta inicialmente pero se ocultará si no es necesario
        self.menu_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Ejecutar configuración inicial después de que todo esté empaquetado
        self.root.after_idle(_configure_canvas)
        
        # Bind mousewheel para scroll
        def _on_mousewheel(event):
            self.menu_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.menu_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Páginas actuales (se crean bajo demanda)
        self.current_page: ttk.Frame | None = None
        self.pages: dict[str, ttk.Frame] = {}

        # Header mejorado con mejor diseño
        header = ttk.Frame(self.menu_content, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 24))
        
        # Título principal con icono
        title_frame = ttk.Frame(header, style="App.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            title_frame, 
            text="📊 Clasificación de Programas SNIES", 
            style="Header.TLabel"
        ).pack(anchor="w")
        
        # Subtítulo mejorado (wraplength dinámico se ajustará en _update_responsive)
        self.subtitle_label = ttk.Label(
            header,
            text="Descarga, normaliza, clasifica programas académicos y gestiona referentes EAFIT.",
            style="SubHeader.TLabel",
            wraplength=800,
            justify="left",
        )
        self.subtitle_label.pack(anchor="w", pady=(4, 0))
        
        # Separador elegante
        separator = ttk.Frame(header, style="Separator.TFrame", height=1)
        separator.pack(fill=tk.X, pady=(16, 0))
        separator.pack_propagate(False)

        # Layout de dos columnas que se expanden para llenar todo el espacio disponible
        main_content = ttk.Frame(self.menu_content, style="App.TFrame")
        main_content.pack(fill=tk.BOTH, expand=True)
        
        # Columna izquierda: Acciones principales y configuración (50% del espacio)
        left_column = ttk.Frame(main_content, style="App.TFrame")
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))
        
        # Columna derecha: Estado del sistema y utilidades (50% del espacio)
        right_column = ttk.Frame(main_content, style="App.TFrame")
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(16, 0))

        # === COLUMNA IZQUIERDA ===
        
        # Card: Acción principal (más destacada y limpia)
        self.primary_action_card = ttk.Frame(left_column, padding=28, style="Card.TFrame")
        self.primary_action_card.pack(fill=tk.X, pady=(0, 24))
        
        # Botón principal más grande y destacado
        primary_btn = ttk.Button(
            self.primary_action_card,
            text="▶️ Ejecutar análisis SNIES (Pipeline)",
            command=self._open_pipeline,
            style="Primary.TButton",
        )
        primary_btn.pack(fill=tk.X, pady=(0, 12))
        bind_tooltip(
            primary_btn,
            "Descarga datos SNIES, normaliza nombres y campos, detecta programas nuevos y "
            "clasifica cada programa con el modelo ML. Genera outputs/Programas.xlsx.",
        )
        
        # Descripción con wraplength dinámico (se actualizará en _update_responsive)
        self.primary_desc_label = ttk.Label(
            self.primary_action_card,
            text="Descarga desde SNIES, normaliza y clasifica programas académicos.",
            style="Muted.TLabel",
            wraplength=400,  # Valor inicial, se actualizará dinámicamente
            justify="left",
        )
        self.primary_desc_label.pack(anchor="w", fill=tk.X)
        
        # Card: Próximos pasos (solo visible cuando falta algo clave)
        self.empty_state_card = ttk.Frame(left_column, padding=20, style="Card.TFrame")
        self.empty_state_title = ttk.Label(
            self.empty_state_card,
            text="📌 Próximos pasos",
            style="SectionTitle.TLabel",
        )
        self.empty_state_title.pack(anchor="w", pady=(0, 8))
        self.empty_state_body = ttk.Label(
            self.empty_state_card,
            text="",
            style="Muted.TLabel",
            wraplength=400,
            justify="left",
        )
        self.empty_state_body.pack(anchor="w", fill=tk.X)
        # Empaquetado condicional desde _refresh_empty_state_hints()
        
        # Card: Otras acciones (más compactas y limpias)
        other_actions_card = ttk.Frame(left_column, padding=28, style="Card.TFrame")
        other_actions_card.pack(fill=tk.X, pady=(0, 24))
        
        other_actions_title = ttk.Label(
            other_actions_card, 
            text="📋 Otras Acciones", 
            style="SectionTitle.TLabel"
        )
        other_actions_title.pack(anchor="w", pady=(0, 16))
        
        self._action_desc_labels = []  # Para actualización responsive
        
        def compact_action_row(
            title: str,
            desc: str,
            cmd,
            icon: str = "",
            style: str = "Secondary.TButton",
            tooltip: str | None = None,
        ):
            row = ttk.Frame(other_actions_card, style="Card.TFrame")
            row.pack(fill=tk.X, pady=(0, 14))
            btn_text = f"{icon} {title}" if icon else title
            btn = ttk.Button(row, text=btn_text, command=cmd, style=style)
            btn.pack(fill=tk.X)
            if tooltip:
                bind_tooltip(btn, tooltip)
            # Descripción más pequeña y discreta con wraplength dinámico
            desc_label = ttk.Label(
                row, 
                text=desc, 
                style="Light.TLabel", 
                wraplength=400,  # Valor inicial, se actualizará dinámicamente
                justify="left",
                font=("Segoe UI", 8)
            )
            desc_label.pack(anchor="w", pady=(4, 0), fill=tk.X)
            self._action_desc_labels.append(desc_label)
        
        compact_action_row(
            "Ajuste manual de emparejamientos",
            "Revisa y corrige ES_REFERENTE y programa EAFIT.",
            self._open_manual,
            icon="✏️",
            tooltip="Abre la tabla de Programas.xlsx para marcar referentes, corregir falsos positivos "
            "y asociar cada programa SNIES al programa EAFIT correspondiente.",
        )
        compact_action_row(
            "Revisión de Áreas",
            "Imputa valores faltantes en ÁREA_DE_CONOCIMIENTO usando IA.",
            self._open_imputacion,
            icon="🤖",
            tooltip="Completa el campo ÁREA_DE_CONOCIMIENTO cuando falta, usando el modelo de imputación. "
            "Los cambios se guardan en Programas.xlsx.",
        )
        compact_action_row(
            "Indicador de oportunidad de portafolio",
            "Pipeline de agregación por categoría y exportación a Estudio_Mercado_Colombia.xlsx.",
            self._open_mercado,
            icon="📊",
            style="Primary.TButton",
            tooltip="Fase 1: base programas×categoría. Fases 2–5: matrículas, OLE, scoring y Excel nacional. "
            "Requiere Programas.xlsx del pipeline SNIES.",
        )
        # NOTA: "Reentrenamiento del modelo" se movió a la sección Utilidades.
        # "Consolidar archivos (Merge)" se oculta temporalmente (no se está usando).
        # Las callbacks self._open_retrain y self._open_merge se conservan por
        # si en el futuro se necesita reactivar cualquiera de las dos acciones.

        # Card: Configuración (más compacta y limpia)
        config_card = ttk.Frame(left_column, padding=28, style="Card.TFrame")
        config_card.pack(fill=tk.X)
        
        ttk.Label(
            config_card, 
            text="⚙️ Configuración", 
            style="SectionTitle.TLabel"
        ).pack(anchor="w", pady=(0, 14))

        self.base_dir: Path | None = None
        self.base_label = ttk.Label(
            config_card,
            text="📁 Carpeta del proyecto: (no configurado)",
            style="Muted.TLabel",
            wraplength=400,  # Valor inicial, se actualizará dinámicamente
            justify="left",
            font=("Segoe UI", 9)
        )
        self.base_label.pack(anchor="w", pady=(0, 14), fill=tk.X)

        ttk.Button(
            config_card,
            text="📂 Cambiar carpeta del proyecto",
            command=self._configure,
            style="Small.TButton",
        ).pack(fill=tk.X)

        # === COLUMNA DERECHA ===
        
        # Card: Estado del Sistema (más compacto y limpio)
        health_card = ttk.Frame(right_column, padding=28, style="Card.TFrame")
        health_card.pack(fill=tk.X, pady=(0, 24))
        
        health_title = ttk.Label(
            health_card, 
            text="💚 Estado del Sistema", 
            style="SectionTitle.TLabel"
        )
        health_title.pack(anchor="w", pady=(0, 14))
        
        # Frame para estados (con mejor formato)
        self.health_frame = ttk.Frame(health_card, style="Card.TFrame")
        self.health_frame.pack(fill=tk.X, pady=(0, 14))
        
        self.health_status_labels = {}
        
        # Botones más compactos en una sola fila con mejor espaciado
        health_btn_frame = ttk.Frame(health_card, style="Card.TFrame")
        health_btn_frame.pack(fill=tk.X)
        
        btn_verify_health = ttk.Button(
            health_btn_frame,
            text="🔍 Verificar",
            command=self._run_health_check,
            style="Small.TButton",
        )
        btn_verify_health.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        bind_tooltip(
            btn_verify_health,
            "Comprueba Internet, archivos en ref/, modelos ML cargables y permisos de escritura en outputs/.",
        )
        btn_repair_health = ttk.Button(
            health_btn_frame,
            text="🔧 Reparar",
            command=self._repair_system,
            style="Small.TButton",
        )
        btn_repair_health.pack(side=tk.LEFT, fill=tk.X, expand=True)
        bind_tooltip(
            btn_repair_health,
            "Según el último Verificar: sugiere pasos o abre reentrenamiento si faltan modelos ML. "
            "No modifica archivos automáticamente salvo que confirmes ir a reentrenar.",
        )
        
        # Ejecutar health check automáticamente al abrir
        self.root.after_idle(lambda: self.root.after(1500, self._run_health_check))
        
        # Card: Utilidades (más compacta y organizada)
        util_card = ttk.Frame(right_column, padding=28, style="Card.TFrame")
        util_card.pack(fill=tk.X, pady=(0, 24))
        
        util_title = ttk.Label(
            util_card, 
            text="🛠️ Utilidades", 
            style="SectionTitle.TLabel"
        )
        util_title.pack(anchor="w", pady=(0, 14))
        
        self.util_btns = ttk.Frame(util_card, style="Card.TFrame")
        self.util_btns.pack(fill=tk.X)
        
        # Botones con textos completos y mejor organización
        util_buttons_data = [
            ("📋 Ver logs", self._open_logs, "Abre logs/pipeline.log con la aplicación predeterminada."),
            ("🔓 Desbloquear", self._unlock_if_needed, "Quita .pipeline.lock si quedó un bloqueo huérfano tras un cierre inesperado."),
            ("📂 Outputs", self._open_outputs, "Abre la carpeta outputs/ del proyecto en el explorador de archivos."),
            (
                "📊 Revisar programas y categorías",
                self._open_base_programas_categorias,
                "Abre el Excel más reciente Base_Programas_Categoria_F1_*.xlsx generado en la Fase 1 del indicador.",
            ),
            (
                "➕ Añadir programas",
                self._open_gestion_programas,
                "Agrega o edita los programas candidatos que se analizan en la Fase 7 (Valorización).",
            ),
            ("🎯 Reentrenar modelo", self._open_retrain, "Página para editar referentes y reentrenar el modelo de clasificación ML."),
            (
                "⚙️ Configuración",
                self._open_configuracion,
                "Ajusta el año de datos activo, verifica archivos disponibles y actualiza SMLMV.",
            ),
        ]
        
        self._util_buttons = []
        for text, cmd, tip in util_buttons_data:
            btn = ttk.Button(self.util_btns, text=text, command=cmd, style="Small.TButton")
            bind_tooltip(btn, tip)
            self._util_buttons.append(btn)
        
        # Organizar botones en grid de 2 columnas con mejor espaciado.
        # Si la cantidad de botones es impar, el último ocupa la fila completa
        # para evitar que quede una celda vacía a su derecha.
        n_btns = len(self._util_buttons)
        last_alone = n_btns % 2 == 1
        for i, btn in enumerate(self._util_buttons):
            row = i // 2
            col = i % 2
            is_last_alone = last_alone and i == n_btns - 1
            pady_bottom = 8 if row < (n_btns - 1) // 2 else 0
            if is_last_alone:
                btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=(0, 0), pady=(0, pady_bottom))
            else:
                padx_right = 8 if col == 0 else 0
                btn.grid(row=row, column=col, sticky="ew", padx=(0, padx_right), pady=(0, pady_bottom))
        
        # Configurar columnas para distribución uniforme
        self.util_btns.columnconfigure(0, weight=1, uniform="util_col")
        self.util_btns.columnconfigure(1, weight=1, uniform="util_col")
        
        # Card: Flujo recomendado (más compacto y visual)
        flow_card = ttk.Frame(right_column, padding=28, style="Card.TFrame")
        flow_card.pack(fill=tk.X)
        
        flow_title = ttk.Label(
            flow_card, 
            text="🔄 Flujo recomendado", 
            style="SectionTitle.TLabel"
        )
        flow_title.pack(anchor="w", pady=(0, 12))
        
        # Flujo más visual y compacto con mejor diseño
        flow_steps = ttk.Frame(flow_card, style="Card.TFrame")
        flow_steps.pack(fill=tk.X)
        
        steps_data = [
            ("1", "Pipeline SNIES"),
            ("2", "Imputar (IA)"),
            ("3", "Ajuste manual"),
            ("4", "Indicador portafolio"),
        ]
        
        for i, (num, text) in enumerate(steps_data):
            step_frame = ttk.Frame(flow_steps, style="Card.TFrame")
            step_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            step_label = ttk.Label(
                step_frame,
                text=f"{num}. {text}",
                style="Muted.TLabel",
                font=("Segoe UI", 9),
                anchor="center"
            )
            step_label.pack(fill=tk.X)
            
            # Agregar flecha entre pasos (excepto después del último)
            if i < len(steps_data) - 1:
                arrow_label = ttk.Label(
                    flow_steps,
                    text="→",
                    style="Muted.TLabel",
                    font=("Segoe UI", 11),
                    foreground=EAFIT["text_muted"]
                )
                arrow_label.pack(side=tk.LEFT, padx=4)
        
        self.flow_label = None  # Ya no se usa el label individual

        # Footer/status mejorado - siempre visible en la parte inferior, FUERA del área scrollable
        # El footer se crea DESPUÉS de todo el contenido para asegurar que quede fijo en la parte inferior
        # y no afecte el área scrollable del contenido central
        # Separador visual antes del footer
        footer_separator = ttk.Frame(self.menu_frame, style="Separator.TFrame", height=1)
        footer_separator.pack_propagate(False)  # Mantener altura fija
        footer_separator.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 0))
        
        # Footer fijo (NO expand, solo fill horizontal) - siempre en la parte inferior
        footer = ttk.Frame(self.menu_frame, style="App.TFrame", padding=(18, 12))
        footer.pack(fill=tk.X, side=tk.BOTTOM)  # Siempre en la parte inferior, sin expand
        
        # Estado mejorado (izquierda)
        status_frame = ttk.Frame(footer, style="App.TFrame")
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_label = ttk.Label(
            status_frame, 
            text="✅ Estado: listo", 
            style="Status.TLabel"
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Botón salir mejorado (derecha, siempre visible)
        exit_btn = ttk.Button(
            footer, 
            text="🚪 Salir", 
            command=self.root.destroy, 
            style="Secondary.TButton"
        )
        exit_btn.pack(side=tk.RIGHT, padx=(12, 0))

        # Responsive: reajuste al redimensionar (wraplength y contenido)
        self._last_width = 0
        self._last_height = 0
        self.root.bind("<Configure>", self._on_configure_resize)
        self.outer.bind("<Configure>", self._on_configure_resize)
        
        # Asegurar que el footer siempre sea visible
        self.root.update_idletasks()
        self._ensure_footer_visible()

        # Cargar configuración y forzar un reajuste inicial cuando la ventana esté dibujada
        self.root.after(100, self._refresh_base_dir)
        self.root.after(350, self._update_responsive)
        self.root.after(500, self._ensure_footer_visible)
    
    def _ensure_footer_visible(self):
        """Asegura que el footer siempre sea visible en la parte inferior."""
        try:
            # Forzar actualización del layout
            self.root.update_idletasks()
            # El footer ya está correctamente posicionado con side=tk.BOTTOM
            # Solo necesitamos asegurar que el contenedor de contenido respete su espacio
        except Exception:
            pass

    def _on_configure_resize(self, event):
        """Cuando se redimensiona la ventana, reajusta wraplength y notifica a las páginas."""
        # Aceptar evento del root o del frame principal
        if event.widget not in (self.root, self.outer):
            return
        self.root.after_idle(self._update_responsive)

    def _update_responsive(self):
        """Actualiza wraplength de labels, botones de utilidades y tablas según el tamaño actual de la ventana."""
        try:
            w_root = self.root.winfo_width()
            h = self.root.winfo_height()
            w_outer = self.outer.winfo_width() if self.outer.winfo_exists() else 0
            # Usar el ancho del área de contenido (outer); si no está mapeado aún, usar root
            w = max(w_root, w_outer) if w_outer > 50 else w_root
            w = max(200, w)
            
            # Calcular ancho disponible para el contenido (usando el canvas real)
            try:
                canvas_width = self.menu_canvas.winfo_width()
                if canvas_width < 10:  # Si el canvas aún no está inicializado
                    canvas_width = w - 40  # Aproximación
            except (tk.TclError, AttributeError):
                canvas_width = w - 40
            
            # Calcular ancho del contenido (considerando padding del menu_content: 24px cada lado = 48px)
            # Y padding entre columnas: 16px cada lado = 32px
            # Y padding interno de cards: ~28px cada lado = 56px por card
            content_width = max(600, canvas_width - 48)  # Padding del menu_content
            wraplen = max(400, content_width - 100)  # Para el subtítulo completo
            
            # Actualizar wraplength de labels específicos
            if abs(w - getattr(self, "_last_width", 0)) >= 5:  # Cambiar más frecuentemente
                self._last_width = w
                
                # Calcular wraplength para columnas (considerando padding entre columnas: 16px cada lado = 32px)
                # Y padding interno de cards: ~28px cada lado = 56px
                column_width = (content_width - 32) // 2  # Dividir entre dos columnas menos padding entre ellas
                # Restar menos padding para que las columnas no sean demasiado estrechas
                wraplen_column = max(250, column_width - 80)  # Menos padding restado para mejor uso del espacio
                
                # Actualizar labels específicos con wraplength dinámico
                if hasattr(self, 'subtitle_label'):
                    try:
                        self.subtitle_label.configure(wraplength=wraplen)
                    except (tk.TclError, AttributeError):
                        pass
                
                # Actualizar descripción de acción principal
                if hasattr(self, 'primary_desc_label'):
                    try:
                        self.primary_desc_label.configure(wraplength=wraplen_column)
                    except (tk.TclError, AttributeError):
                        pass
                
                if hasattr(self, 'base_label'):
                    try:
                        self.base_label.configure(wraplength=wraplen_column)
                    except (tk.TclError, AttributeError):
                        pass
                
                # Actualizar descripciones de acciones (en columnas)
                if hasattr(self, '_action_desc_labels'):
                    for label in self._action_desc_labels:
                        try:
                            label.configure(wraplength=wraplen_column)
                        except (tk.TclError, AttributeError):
                            pass

                if hasattr(self, "empty_state_body"):
                    try:
                        self.empty_state_body.configure(wraplength=wraplen_column)
                    except (tk.TclError, AttributeError):
                        pass
                
                # Actualizar labels de estado del sistema si existen
                if hasattr(self, 'health_status_labels'):
                    for check_name, (label, ok) in self.health_status_labels.items():
                        try:
                            label.configure(wraplength=wraplen_column)
                        except (tk.TclError, AttributeError):
                            pass
                
                # Los botones de utilidades ya están en grid de 2 columnas, no necesitan reorganización
            
            # Asegurar que el footer sea visible
            self._ensure_footer_visible()
            
            # Notificar a la página actual para que actualice tablas (altura responsive)
            if self.current_page and hasattr(self.current_page, "_on_resize"):
                if abs(h - getattr(self, "_last_height", 0)) >= 20:
                    self._last_height = h
                    self.current_page._on_resize(w, h)
        except (tk.TclError, AttributeError):
            pass

    def _relayout_util_buttons(self, w: int):
        """Reorganiza los botones de Utilidades en una o múltiples filas según el ancho."""
        if not hasattr(self, "_util_buttons") or not self._util_buttons:
            return
        
        for i, btn in enumerate(self._util_buttons):
            try:
                btn.grid_forget()
            except tk.TclError:
                pass
        
        # Calcular cuántos botones caben en una fila según el ancho disponible
        # Cada botón necesita aproximadamente 140-180px de ancho (dependiendo del texto)
        # Usar un cálculo más conservador para evitar cortes
        estimated_button_width = 160  # Ancho estimado por botón
        buttons_per_row = max(2, min(5, int((w - 40) / estimated_button_width)))  # -40 para padding
        
        # Si hay más botones de los que caben, usar múltiples filas
        if len(self._util_buttons) > buttons_per_row:
            # Usar múltiples filas
            for i, btn in enumerate(self._util_buttons):
                row = i // buttons_per_row
                col = i % buttons_per_row
                padx_left = 0 if col == 0 else 8
                padx_right = 0
                pady_top = 0 if row == 0 else 6
                btn.grid(
                    row=row, 
                    column=col, 
                    sticky="w", 
                    padx=(padx_left, padx_right), 
                    pady=(pady_top, 0)
                )
        else:
            # Una sola fila - distribuir uniformemente
            for i, btn in enumerate(self._util_buttons):
                padx_left = 0 if i == 0 else 8
                btn.grid(
                    row=0, 
                    column=i, 
                    sticky="w", 
                    padx=(padx_left, 0)
                )
        
        # Actualizar el frame de botones para que se ajuste
        try:
            self.util_btns.update_idletasks()
        except:
            pass

    def _all_children(self, parent):
        """Generador recursivo de todos los hijos de un widget."""
        try:
            for c in parent.winfo_children():
                yield c
                yield from self._all_children(c)
        except (tk.TclError, AttributeError):
            pass

    def _refresh_base_dir(self):
        # No pedir carpeta al abrir el menú; solo mostrar estado.
        # Operaciones ligeras primero (sin I/O pesado)
        bd = get_configured_base_dir()
        if bd:
            self.base_dir = bd
            self.base_label.config(text=f"Carpeta del proyecto: {bd}")
            if hasattr(self, "status_label"):
                self.status_label.config(text="Estado: listo")
        else:
            self.base_dir = None
            self.base_label.config(text="Carpeta del proyecto: (no configurado)")
            if hasattr(self, "status_label"):
                self.status_label.config(text="Estado: requiere configuración")

        # Operaciones I/O más pesadas se hacen después (async)
        # Esto evita bloquear el inicio de la aplicación
        self.root.after(200, self._refresh_status_async)
    
    def _refresh_status_async(self):
        """Actualiza el estado de forma asíncrona (no bloquea el inicio)."""
        try:
            # Indicar lock (pipeline corriendo o lock huérfano)
            lock_file = get_pipeline_lock_file()
            age = get_lock_age_seconds(lock_file)
            if age is not None and hasattr(self, "status_label"):
                if age > LOCK_STALE_SECONDS:
                    self.status_label.config(text="Estado: lock detectado (posible cierre inesperado)")
                else:
                    self.status_label.config(text="Estado: pipeline en ejecución")
            else:
                # Si hay Programas.xlsx, mostrar fecha de actualización
                try:
                    from etl.normalizacion import ARCHIVO_PROGRAMAS  # Lazy import
                    p = ARCHIVO_PROGRAMAS
                    if p.exists() and hasattr(self, "status_label"):
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                        self.status_label.config(text=f"Estado: listo (Programas.xlsx actualizado: {ts})")
                except Exception:
                    pass
        except Exception:
            # Si falla, no bloquear la aplicación
            pass
        try:
            self._refresh_empty_state_hints()
        except Exception:
            pass

    def _refresh_empty_state_hints(self) -> None:
        """Muestra u oculta la tarjeta «Próximos pasos» según archivos clave del proyecto."""
        if not hasattr(self, "empty_state_card"):
            return
        try:
            bd = get_configured_base_dir()
            lines: list[str] = []
            if not bd:
                lines.append("• Configura la carpeta del proyecto (sección «Configuración», más abajo en esta pantalla).")
            else:
                from etl.normalizacion import ARCHIVO_PROGRAMAS

                if not ARCHIVO_PROGRAMAS.exists():
                    lines.append(
                        "• Ejecuta «Pipeline SNIES» (botón principal arriba) para generar outputs/Programas.xlsx."
                    )
                else:
                    try:
                        from etl.config import ESTUDIO_MERCADO_DIR

                        if ESTUDIO_MERCADO_DIR.exists():
                            hay_base = any(ESTUDIO_MERCADO_DIR.glob("Base_Programas_Categoria_F1_*.xlsx"))
                        else:
                            hay_base = False
                    except Exception:
                        hay_base = False
                    if not hay_base:
                        lines.append(
                            "• Abre «Indicador de oportunidad de portafolio» y ejecuta la Fase 1 para generar "
                            "Base_Programas_Categoria_F1_*.xlsx (programas y categorías)."
                        )
            if not lines:
                self.empty_state_card.pack_forget()
                return
            body = "\n".join(lines) + "\n\nCuando completes estos pasos, esta tarjeta se ocultará sola."
            self.empty_state_body.config(text=body)
            try:
                self.empty_state_card.pack(fill=tk.X, pady=(0, 16), after=self.primary_action_card)
            except tk.TclError:
                self.empty_state_card.pack(fill=tk.X, pady=(0, 16))
            self.root.after_idle(self._update_responsive)
        except Exception:
            pass

    def _configure(self):
        # Forzar selección de carpeta
        selected_dir = filedialog.askdirectory(title="Seleccionar carpeta raíz del proyecto", initialdir=str(Path.home()), parent=self.root)
        if not selected_dir:
            return
        p = Path(selected_dir)
        if not set_base_dir(p):
            messagebox.showerror("Error", "No se pudo guardar la configuración.", parent=self.root)
            return
        try:
            update_paths_for_base_dir(p)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo aplicar la configuración:\n\n{exc}", parent=self.root)
            return
        self.base_dir = p
        self.base_label.config(text=f"Carpeta del proyecto: {p}")
        if hasattr(self, "status_label"):
            self.status_label.config(text="Estado: listo")
        # Re-ejecutar health check después de configurar
        self.root.after(500, self._run_health_check)
        self.root.after(600, self._refresh_empty_state_hints)

    def _run_health_check(self):
        """Ejecuta diagnóstico del sistema y muestra resultados."""
        # Limpiar frame de health
        for widget in self.health_frame.winfo_children():
            widget.destroy()
        
        checks = {}
        
        # 1. Conexión a Internet
        try:
            import urllib.request
            urllib.request.urlopen("https://www.google.com", timeout=5)
            checks["internet"] = (True, "Conexión a Internet: OK")
        except Exception:
            checks["internet"] = (False, "Conexión a Internet: ❌ No disponible (necesario para descarga SNIES)")
        
        # 2. Archivos base (ref/)
        try:
            from etl.config import REF_DIR, get_archivo_referentes, get_archivo_catalogo_eafit
            ref_dir_ok = REF_DIR.exists() and REF_DIR.is_dir()
            referentes_ok = get_archivo_referentes().exists()
            catalogo_ok = get_archivo_catalogo_eafit().exists()
            
            if ref_dir_ok and referentes_ok and catalogo_ok:
                checks["archivos_ref"] = (True, "Archivos de referencia (ref/): OK")
            else:
                msg = "Archivos de referencia (ref/): ❌ Faltan archivos"
                if not referentes_ok:
                    msg += " (referentesUnificados)"
                if not catalogo_ok:
                    msg += " (catalogoOfertasEAFIT)"
                checks["archivos_ref"] = (False, msg)
        except Exception as e:
            checks["archivos_ref"] = (False, f"Archivos de referencia: ❌ Error: {e}")
        
        # 3. Modelos ML
        try:
            from etl.clasificacionProgramas import MODELO_CLASIFICADOR, MODELO_EMBEDDINGS_OBJ, ENCODER_PROGRAMAS_EAFIT
            modelos_ok = all([
                MODELO_CLASIFICADOR.exists(),
                MODELO_EMBEDDINGS_OBJ.exists(),
                ENCODER_PROGRAMAS_EAFIT.exists()
            ])
            if modelos_ok:
                # Intentar cargar para verificar integridad
                try:
                    from etl.clasificacionProgramas import cargar_modelos
                    cargar_modelos()
                    checks["modelos"] = (True, "Modelos ML: OK (cargados correctamente)")
                except Exception as e:
                    checks["modelos"] = (False, f"Modelos ML: ⚠️ Existen pero están corruptos: {e}")
            else:
                checks["modelos"] = (False, "Modelos ML: ❌ No encontrados (ejecuta reentrenamiento)")
        except Exception as e:
            checks["modelos"] = (False, f"Modelos ML: ❌ Error: {e}")
        
        # 4. Permisos de escritura en outputs/
        try:
            from etl.config import OUTPUTS_DIR
            test_file = OUTPUTS_DIR / ".test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()
                checks["permisos"] = (True, "Permisos de escritura (outputs/): OK")
            except PermissionError:
                checks["permisos"] = (False, "Permisos de escritura (outputs/): ❌ Sin permisos de escritura")
            except Exception as e:
                checks["permisos"] = (False, f"Permisos de escritura: ❌ Error: {e}")
        except Exception as e:
            checks["permisos"] = (False, f"Permisos de escritura: ❌ Error: {e}")
        
        # Mostrar resultados de forma más limpia y compacta
        # Obtener ancho disponible para wraplength dinámico
        try:
            canvas_width = self.menu_canvas.winfo_width() if hasattr(self, 'menu_canvas') else 800
            content_width = max(600, canvas_width - 48)
            column_width = (content_width - 32) // 2
            health_wraplength = max(200, column_width - 80)
        except (tk.TclError, AttributeError):
            health_wraplength = 300  # Fallback
        
        for check_name, (ok, msg) in checks.items():
            # Simplificar mensajes para que sean más cortos
            if ok:
                # Mensajes OK más cortos
                if "Internet" in msg:
                    display_msg = "🌐 Internet: OK"
                elif "referencia" in msg.lower():
                    display_msg = "📁 Archivos ref/: OK"
                elif "Modelos" in msg:
                    display_msg = "🤖 Modelos ML: OK"
                elif "Permisos" in msg:
                    display_msg = "✍️ Permisos: OK"
                else:
                    display_msg = msg.split(":")[0] + ": OK" if ":" in msg else msg
            else:
                # Mensajes de error más cortos
                if "Internet" in msg:
                    display_msg = "🌐 Internet: ❌"
                elif "referencia" in msg.lower():
                    display_msg = "📁 Archivos ref/: ❌"
                elif "Modelos" in msg:
                    display_msg = "🤖 Modelos ML: ❌"
                elif "Permisos" in msg:
                    display_msg = "✍️ Permisos: ❌"
                else:
                    display_msg = msg.split(":")[0] + ": ❌" if ":" in msg else msg
            
            color = EAFIT["success"] if ok else EAFIT["danger"]
            label = ttk.Label(
                self.health_frame,
                text=display_msg,
                foreground=color,
                style="Muted.TLabel",
                font=("Segoe UI", 9),
                wraplength=health_wraplength,  # Agregar wraplength dinámico
                justify="left"
            )
            label.pack(anchor="w", pady=3, fill=tk.X)
            self.health_status_labels[check_name] = (label, ok)
    
    def _repair_system(self):
        """Intenta reparar problemas detectados en el health check."""
        problemas = []
        soluciones = []
        
        # Verificar cada check
        for check_name, (label_widget, ok) in self.health_status_labels.items():
            if not ok:
                problemas.append(check_name)
        
        if not problemas:
            messagebox.showinfo("Sistema OK", "No se detectaron problemas que requieran reparación.", parent=self.root)
            return
        
        # Intentar reparar cada problema
        if "archivos_ref" in problemas:
            soluciones.append("Verifica que los archivos en ref/ existan y tengan el formato correcto.")
        
        if "modelos" in problemas:
            respuesta = messagebox.askyesno(
                "Reparar Modelos",
                "Los modelos ML no están disponibles o están corruptos.\n\n"
                "¿Deseas ir a la página de reentrenamiento para entrenar nuevos modelos?",
                parent=self.root
            )
            if respuesta:
                self._open_retrain()
                return
        
        if "permisos" in problemas:
            soluciones.append("Verifica que tengas permisos de escritura en la carpeta outputs/.")
            soluciones.append("Cierra Excel/Power BI si tienen archivos abiertos en outputs/.")
        
        if "internet" in problemas:
            soluciones.append("Verifica tu conexión a Internet.")
            soluciones.append("El sistema necesita Internet para descargar datos desde SNIES.")
        
        mensaje = "Problemas detectados:\n\n"
        mensaje += "\n".join(f"• {p}" for p in problemas)
        mensaje += "\n\nSoluciones sugeridas:\n\n"
        mensaje += "\n".join(f"• {s}" for s in soluciones)
        
        messagebox.showinfo("Reparación del Sistema", mensaje, parent=self.root)

    def _show_page(self, page_name: str, page_class, *args, **kwargs):
        """Muestra una página específica, ocultando el menú y otras páginas."""
        # Ocultar menú
        self.menu_frame.pack_forget()
        
        # Ocultar página actual si existe
        if self.current_page:
            self.current_page.pack_forget()
        
        # Mostrar content_container (si estaba oculto)
        self.content_container.pack(fill=tk.BOTH, expand=True)
        
        # Crear o reutilizar página
        if page_name not in self.pages:
            self.pages[page_name] = page_class(self.content_container, on_back=self._show_menu, *args, **kwargs)
        
        self.current_page = self.pages[page_name]
        self.current_page.pack(fill=tk.BOTH, expand=True)
        
        # Mínimos por página (la ventana conserva el tamaño actual; el usuario puede maximizar o reducir)
        if page_name == "pipeline":
            self.root.minsize(700, 500)
        elif page_name in ("manual", "retrain"):
            self.root.minsize(900, 600)
        elif page_name == "merge":
            self.root.minsize(700, 450)
        elif page_name == "imputacion":
            self.root.minsize(700, 500)
        elif page_name == "mercado":
            self.root.minsize(700, 550)
        elif page_name == "mercado_results":
            self.root.minsize(1000, 600)
        
        self.root.update_idletasks()
        
        # Llamar _on_resize después de mostrar la página para ajustar elementos responsive
        if hasattr(self.current_page, "_on_resize"):
            try:
                w = self.root.winfo_width()
                h = self.root.winfo_height()
                self.current_page._on_resize(w, h)
            except (tk.TclError, AttributeError):
                pass
        # Forzar un reajuste responsive al cambiar de página (tabla y wraplength)
        self.root.after(50, self._update_responsive)
        self.root.after(200, self._update_responsive)  # segunda pasada tras layout completo
    
    def _show_menu(self):
        """Vuelve al menú principal, ocultando la página actual."""
        # Ocultar página actual si existe
        if self.current_page:
            self.current_page.pack_forget()
        self.current_page = None
        
        # Ocultar content_container
        self.content_container.pack_forget()
        
        # Mostrar menú
        self.menu_frame.pack(fill=tk.BOTH, expand=True)
        
        self.root.minsize(800, 500)
        self._refresh_base_dir()
        self.root.update_idletasks()
        self.root.after(200, self._refresh_empty_state_hints)

    def _open_pipeline(self):
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        self._show_page("pipeline", PipelinePage)

    def _open_manual(self):
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        self._show_page("manual", ManualReviewPage)

    def _open_retrain(self):
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        self._show_page("retrain", RetrainPage)

    def _open_merge(self):
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        self._show_page("merge", MergePage)

    def _open_imputacion(self):
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        # ImputationPage está definida en este mismo archivo
        self._show_page("imputacion", ImputationPage)

    def _open_mercado(self):
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        self._show_page("mercado", MercadoPipelinePage)

    def _open_configuracion(self) -> None:
        """Abre el diálogo de configuración del sistema."""
        dlg = ConfiguracionDialog(self.root)
        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)

    def _open_gestion_programas(self) -> None:
        dlg = GestionProgramasDialog(self.root)
        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)

    def _open_logs(self):
        base = get_configured_base_dir()
        if not base:
            safe_messagebox_error("Error", "Configura primero la carpeta del proyecto.", parent=self.root)
            return
        log_path = base / "logs" / "pipeline.log"
        if not log_path.exists():
            safe_messagebox_error("Atención", f"No existe el log aún:\n{log_path}", parent=self.root)
            return
        try:
            _open_text_file(log_path)
        except Exception:
            # fallback: abrir con app por defecto
            try:
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            except Exception as exc:
                safe_messagebox_error("Error", str(exc), parent=self.root)

    def _unlock_if_needed(self):
        lock_file = get_pipeline_lock_file()
        if not lock_file.exists():
            messagebox.showinfo("OK", "No hay lock activo.", parent=self.root)
            return
        age = get_lock_age_seconds(lock_file)
        if age is None:
            age = 0
        if age < LOCK_STALE_SECONDS:
            messagebox.showwarning(
                "Atención",
                "Parece que el pipeline está en ejecución.\n\n"
                "Solo desbloquea si estás seguro de que NO hay un proceso corriendo.",
                parent=self.root,
            )
        if _ask_yes_no("Confirmar", "¿Eliminar el lock para desbloquear la edición?"):
            try:
                lock_file.unlink()
                messagebox.showinfo("OK", "Lock eliminado.", parent=self.root)
                self._refresh_base_dir()
            except Exception as exc:
                safe_messagebox_error("Error", f"No se pudo eliminar el lock: {exc}", parent=self.root)

    def _open_outputs(self):
        base = get_configured_base_dir()
        if not base:
            safe_messagebox_error("Error", "Configura primero la carpeta del proyecto.", parent=self.root)
            return
        out_dir = base / "outputs"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            _open_default_app(out_dir)
        except Exception as exc:
            safe_messagebox_error("Error", str(exc), parent=self.root)

    def _open_programas(self):
        from etl.normalizacion import ARCHIVO_PROGRAMAS  # Lazy import
        
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        p = ARCHIVO_PROGRAMAS
        if not p.exists():
            safe_messagebox_error("Atención", "Aún no existe outputs/Programas.xlsx. Ejecuta el análisis SNIES.", parent=self.root)
            return
        try:
            _open_in_excel(p)
        except Exception as exc:
            safe_messagebox_error("Error", str(exc), parent=self.root)

    def _open_base_programas_categorias(self):
        """Abre el archivo Base_Programas_Categoria_F1_*.xlsx más reciente
        generado por la Fase 1 del pipeline de estudio de mercado.
        """
        if not ensure_base_dir(self.root, prompt_if_missing=True):
            return
        try:
            from etl.config import ESTUDIO_MERCADO_DIR  # Lazy import
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo localizar la carpeta de estudio de mercado:\n{exc}", parent=self.root)
            return

        if not ESTUDIO_MERCADO_DIR.exists():
            safe_messagebox_error(
                "Atención",
                "Aún no existe la carpeta de estudio de mercado.\n"
                "Ejecuta primero la Fase 1 del Indicador de oportunidad de portafolio.",
                parent=self.root,
            )
            return

        candidatos = sorted(
            ESTUDIO_MERCADO_DIR.glob("Base_Programas_Categoria_F1_*.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidatos:
            safe_messagebox_error(
                "Atención",
                "No se encontró ningún archivo Base_Programas_Categoria_F1_*.xlsx.\n"
                "Ejecuta primero la Fase 1 del Indicador de oportunidad de portafolio.",
                parent=self.root,
            )
            return

        try:
            _open_in_excel(candidatos[0])
        except Exception as exc:
            safe_messagebox_error("Error", str(exc), parent=self.root)
    
class PipelinePage(ttk.Frame):
    """Interfaz gráfica para el pipeline de análisis SNIES."""
    
    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        self.root = parent.winfo_toplevel()
        
        # Import lazy de ARCHIVO_PROGRAMAS (se usa en validaciones)
        from etl.normalizacion import ARCHIVO_PROGRAMAS
        self.ARCHIVO_PROGRAMAS = ARCHIVO_PROGRAMAS
        
        # Estado del pipeline
        self.is_running = False
        self.base_dir = None
        self.cancel_event = threading.Event()  # Evento para cancelar la ejecución
        
        # Configurar el estilo
        self._setup_ui()
        
        # Verificar configuración inicial
        self._check_initial_config()

    def _on_resize(self, w: int, h: int) -> None:
        """Responsive: ajusta wraplengths dinámicamente."""
        # Ajustar wraplength del label de descripción
        if hasattr(self, 'pipeline_desc_label'):
            wraplength = max(400, w - 100)
            self.pipeline_desc_label.config(wraplength=wraplength)

    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        main_frame = ttk.Frame(self, padding=20, style="Page.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título y navegación mejorado
        title_frame = ttk.Frame(main_frame, style="Page.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame, text="▶️ Pipeline de Análisis SNIES", style="Header.TLabel").pack(side=tk.LEFT)
        if self.on_back:
            self.btn_pipeline_back = ttk.Button(
                title_frame,
                text="← Volver al menú",
                command=lambda: self.on_back() if self.on_back else None,
                style="Back.TButton",
            )
            self.btn_pipeline_back.pack(side=tk.RIGHT)
        
        # Card: Carpeta del proyecto mejorada
        dir_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        dir_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(dir_card, text="📁 Carpeta del proyecto", style="SectionTitle.TLabel").pack(anchor="w")
        self.dir_label = ttk.Label(dir_card, text="No configurado", style="Muted.TLabel")
        self.dir_label.pack(anchor="w", pady=(8, 10))
        self.btn_change_dir = ttk.Button(dir_card, text="📂 Cambiar carpeta", command=self._select_base_directory, style="Secondary.TButton")
        self.btn_change_dir.pack(anchor="w")
        
        # Card: Ejecución mejorada
        run_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        run_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(run_card, text="🚀 Ejecutar", style="SectionTitle.TLabel").pack(anchor="w")
        self.pipeline_desc_label = ttk.Label(run_card, text="Descarga SNIES, normaliza, detecta nuevos y clasifica. Puede tardar varios minutos.", style="Muted.TLabel")
        self.pipeline_desc_label.pack(anchor="w", pady=(6, 12), fill=tk.X)
        self.last_success_label = ttk.Label(run_card, text="Última ejecución exitosa: —", style="Muted.TLabel")
        self.last_success_label.pack(anchor="w", pady=(0, 8))
        btn_frame = ttk.Frame(run_card, style="Card.TFrame")
        btn_frame.pack(fill=tk.X)
        self.btn_execute = ttk.Button(btn_frame, text="▶️ Ejecutar pipeline", command=self._on_execute_clicked, state=tk.DISABLED, style="Primary.TButton")
        self.btn_execute.pack(side=tk.LEFT)
        self.btn_cancel = ttk.Button(btn_frame, text="⏹️ Cancelar", command=self._on_cancel_clicked, state=tk.DISABLED, style="Danger.TButton")
        self.btn_cancel.pack(side=tk.LEFT, padx=(10, 0))
        self.btn_validar = ttk.Button(btn_frame, text="🔍 Validar entorno", command=self._on_validar_entorno, style="Secondary.TButton")
        self.btn_validar.pack(side=tk.LEFT, padx=(10, 0))
        self.progress_label = ttk.Label(btn_frame, text="Progreso: listo", style="Status.TLabel")
        self.progress_label.pack(side=tk.LEFT, padx=(16, 0))
        self.progress = ttk.Progressbar(run_card, mode="determinate", maximum=7, value=0)
        self.progress.pack(fill=tk.X, pady=(12, 0))
        
        # Card: Estado y mensajes mejorada
        status_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        status_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        ttk.Label(status_card, text="📋 Estado y mensajes", style="SectionTitle.TLabel").pack(anchor="w")
        self.status_label = ttk.Label(status_card, text="Listo. Configure la carpeta para habilitar la ejecución.", style="Muted.TLabel")
        self.status_label.pack(anchor="w", pady=(6, 8))
        scrollbar = ttk.Scrollbar(status_card)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.messages_text = tk.Text(
            status_card,
            height=10,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            font=("Consolas", 9),
            bg=EAFIT["card_bg"],
            fg=EAFIT["text"],
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.messages_text.yview)
    
    def _refresh_last_success_label(self):
        """Actualiza la etiqueta de última ejecución exitosa desde config."""
        try:
            from etl.config import get_last_success
            iso_ts, dur_min = get_last_success()
            if iso_ts:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
                    text = dt.strftime("Última ejecución exitosa: %d/%m/%Y %H:%M")
                    if dur_min is not None:
                        text += f" ({dur_min:.1f} min)"
                except Exception:
                    text = f"Última ejecución exitosa: {iso_ts[:16]}"
            else:
                text = "Última ejecución exitosa: —"
            self.last_success_label.config(text=text)
        except Exception:
            self.last_success_label.config(text="Última ejecución exitosa: —")

    def _on_validar_entorno(self):
        """Ejecuta validación del entorno y muestra el resultado en un messagebox."""
        ok, mensajes = validar_entorno_pipeline()
        if ok:
            messagebox.showinfo("Validar entorno", "\n".join(mensajes), parent=self.root)
        else:
            messagebox.showwarning(
                "Validar entorno",
                "El entorno no está listo:\n\n" + "\n".join(f"• {m}" for m in mensajes),
                parent=self.root
            )

    def _check_initial_config(self):
        """
        Detecta automáticamente la carpeta del proyecto desde la ubicación
        del ejecutable. No muestra diálogos de selección — funciona para
        cualquier usuario que tenga el proyecto sincronizado en OneDrive.
        """
        base_dir = get_configured_base_dir()
        self._refresh_last_success_label()

        if not base_dir or not base_dir.exists():
            self._log_message("❌ Error: no se pudo detectar la carpeta del proyecto.")
            self._log_message("Verifique que SniesManager.exe esté junto a las carpetas ref/, outputs/, models/.")
            messagebox.showerror(
                "Error de configuración",
                "No se pudo detectar la carpeta del proyecto.\n\n"
                "Verifique que SniesManager.exe esté en la carpeta raíz "
                "del proyecto, junto a las carpetas:\n"
                "  - ref/\n  - outputs/\n  - models/\n  - docs/\n\n"
                "No mueva el ejecutable fuera de esa carpeta.\n"
                "Si el problema persiste, use el botón 'Cambiar carpeta'.",
            )
        else:
            self.base_dir = base_dir
            self._update_dir_label()
            self.btn_execute.config(state=tk.NORMAL)
            self._refresh_last_success_label()
            self._log_message(f"✓ Carpeta del proyecto detectada automáticamente: {base_dir}")
            self._log_message("Listo para ejecutar el pipeline")
    
    def _select_base_directory(self):
        """Abre un diálogo para seleccionar la carpeta raíz del proyecto."""
        selected_dir = filedialog.askdirectory(
            title="Seleccionar carpeta raíz del proyecto",
            initialdir=str(Path.home())
        )
        
        if selected_dir:
            base_dir = Path(selected_dir)
            
            # Verificar que sea una carpeta válida
            if not self._validate_base_directory(base_dir):
                messagebox.showerror(
                    "Error de Validación",
                    "La carpeta seleccionada no contiene las estructuras necesarias.\n\n"
                    "Asegúrese de que la carpeta contenga (o pueda contener):\n"
                    "- ref/\n"
                    "- models/\n"
                    "- docs/\n"
                )
                return
            
            # Guardar la configuración
            if set_base_dir(base_dir):
                self.base_dir = base_dir
                self._update_dir_label()
                self.btn_execute.config(state=tk.NORMAL)
                self._log_message(f"✓ Carpeta configurada: {base_dir}")
                self._log_message("Listo para ejecutar el pipeline")
            else:
                messagebox.showerror("Error", "No se pudo guardar la configuración.")
    
    def _validate_base_directory(self, base_dir: Path) -> bool:
        """
        Valida que el directorio base tenga la estructura esperada.
        
        Args:
            base_dir: Directorio a validar
            
        Returns:
            True si es válido, False en caso contrario
        """
        # Las carpetas ref/, models/, docs/ deben existir o poder crearse
        required_dirs = ["ref", "models", "docs"]
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            # Si no existe, intentar crearlo (solo para verificar permisos)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(exist_ok=True)
                except Exception:
                    return False
        return True
    
    def _update_dir_label(self):
        """Actualiza la etiqueta que muestra el directorio base."""
        if self.base_dir:
            # Mostrar ruta truncada si es muy larga
            dir_str = str(self.base_dir)
            if len(dir_str) > 60:
                dir_str = "..." + dir_str[-57:]
            self.dir_label.config(text=dir_str, foreground=EAFIT["text"])
        else:
            self.dir_label.config(text="No configurado", foreground=EAFIT["text_muted"])
    
    def _log_message(self, message: str):
        """Agrega un mensaje al área de texto."""
        self.messages_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.messages_text.see(tk.END)
        self.messages_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _update_status(self, status: str, color: str = "black"):
        """Actualiza el estado mostrado en la interfaz. Usa colorimetría EAFIT."""
        color_map = {
            "black": EAFIT["text"],
            "green": EAFIT["success"],
            "red": EAFIT["danger"],
            "orange": EAFIT["warning"],
            "gray": EAFIT["text_muted"],
        }
        fg = color_map.get(color, color)
        self.status_label.config(text=status, foreground=fg)
        self.root.update_idletasks()
    
    def _on_execute_clicked(self):
        """Maneja el evento de clic en el botón de ejecutar."""
        if self.is_running:
            messagebox.showwarning("Atención", "El pipeline ya se está ejecutando.", parent=self.root)
            return
        
        if not self.base_dir:
            messagebox.showerror("Error", "Debe configurar la carpeta del proyecto primero.", parent=self.root)
            return

        # Pre-chequeos de estabilidad:
        # - lock activo
        lock = get_pipeline_lock_file()
        age = get_lock_age_seconds(lock)
        if age is not None and age < LOCK_STALE_SECONDS:
            messagebox.showwarning(
                "Atención",
                "Se detectó que el pipeline ya está en ejecución (lock activo).\n\n"
                "Espera a que termine antes de iniciar otro proceso.",
                parent=self.root,
            )
            return
        if age is not None and age >= LOCK_STALE_SECONDS:
            # lock viejo: permitir desbloquear
            if _ask_yes_no(
                "Lock detectado",
                "Se detectó un lock antiguo (posible cierre inesperado).\n\n"
                "¿Deseas eliminar el lock para continuar?",
            ):
                try:
                    lock.unlink()
                except Exception:
                    pass

        # - archivo Programas.xlsx abierto/bloqueado
        if self.ARCHIVO_PROGRAMAS.exists() and not can_write_file(self.ARCHIVO_PROGRAMAS):
            safe_messagebox_error("Error", explain_file_in_use(), parent=self.root)
            return
        
        # Confirmar ejecución
        result = messagebox.askyesno(
            "Confirmar Ejecución",
            "¿Desea ejecutar el pipeline ahora?\n\n"
            "Este proceso puede tardar varios minutos.",
            parent=self.root
        )
        
        if result:
            self._execute_pipeline()
    
    def _execute_pipeline(self):
        """Ejecuta el pipeline en un hilo separado."""
        self.is_running = True
        self.cancel_event.clear()  # Resetear el evento de cancelación
        self.btn_execute.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.btn_validar.config(state=tk.DISABLED)
        self.btn_change_dir.config(state=tk.DISABLED)
        try:
            self.btn_pipeline_back.config(state=tk.DISABLED)
        except (tk.TclError, AttributeError):
            pass
        try:
            self.root.config(cursor="watch")
        except tk.TclError:
            pass
        self._update_status("Procesando...", "orange")
        self._log_message("=" * 50)
        self._log_message("Iniciando ejecución del pipeline...")
        self.progress.config(value=0)
        self.progress_label.config(text="Progreso: iniciando...", foreground=EAFIT["text_muted"])
        
        # Ejecutar en un hilo separado para no bloquear la GUI
        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()
    
    def _on_cancel_clicked(self):
        """Maneja el clic en el botón Cancelar."""
        if not self.is_running:
            return
        
        result = messagebox.askyesno(
            "Cancelar ejecución",
            "¿Está seguro de que desea cancelar la ejecución del pipeline?\n\n"
            "Los cambios realizados hasta el momento se perderán.",
            parent=self.root
        )
        
        if result:
            self.cancel_event.set()
            self._log_message("[CANCELADO] Cancelación solicitada por el usuario...")
            self._update_status("Cancelando...", "orange")
    
    def _run_pipeline_thread(self):
        """Ejecuta el pipeline en un hilo separado."""
        try:
            # Actualizar rutas para usar el base_dir configurado
            update_paths_for_base_dir(self.base_dir)
            
            # Ejecutar el pipeline
            def progress_cb(stage_idx: int, stage_name: str, status: str):
                # status: "start" | "done"
                def apply():
                    if status == "start":
                        self.progress_label.config(text=f"Progreso: {stage_name}...", foreground=EAFIT["text_muted"])
                        self.progress.config(value=stage_idx)
                    elif status == "done":
                        self.progress_label.config(text=f"Progreso: {stage_name} ✓", foreground=EAFIT["success"])
                        self.progress.config(value=stage_idx + 1)
                self.root.after(0, apply)

            resultado = run_pipeline(
                self.base_dir,
                log_callback=self._log_message,
                progress_callback=progress_cb,
                cancel_event=self.cancel_event
            )
            
            # Verificar si fue cancelado (verificar ANTES de actualizar UI para evitar race condition)
            was_cancelled = self.cancel_event.is_set()
            
            if was_cancelled:
                self.root.after(0, self._on_pipeline_error, "Cancelado por el usuario")
            else:
                # Actualizar UI en el hilo principal
                self.root.after(0, self._on_pipeline_completed, resultado == 0)
            
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            self.root.after(0, self._on_pipeline_error, error_msg)
    
    def _on_pipeline_completed(self, success: bool):
        """Maneja la finalización del pipeline."""
        self.is_running = False
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.btn_validar.config(state=tk.NORMAL)
        self.btn_change_dir.config(state=tk.NORMAL)
        try:
            self.btn_pipeline_back.config(state=tk.NORMAL)
        except (tk.TclError, AttributeError):
            pass
        try:
            self.root.config(cursor="")
        except tk.TclError:
            pass
        
        if success:
            self._refresh_last_success_label()
            self._update_status("Completado", "green")
            self._log_message("=" * 50)
            self._log_message("✓ Pipeline completado exitosamente")
            messagebox.showinfo(
                "Éxito",
                "El pipeline se ejecutó correctamente.\n\n"
                f"Los archivos se guardaron en:\n{self.base_dir / 'outputs'}",
                parent=self.root
            )
        else:
            self._update_status("Error", "red")
            self._log_message("=" * 50)
            self._log_message("✗ El pipeline finalizó con errores")
            messagebox.showerror(
                "Error",
                "El pipeline finalizó con errores.\n\n"
                "Revise los mensajes para más detalles.",
                parent=self.root
            )
    
    def _on_pipeline_error(self, error_msg: str):
        """Maneja errores durante la ejecución del pipeline."""
        self.is_running = False
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.btn_validar.config(state=tk.NORMAL)
        self.btn_change_dir.config(state=tk.NORMAL)
        try:
            self.btn_pipeline_back.config(state=tk.NORMAL)
        except (tk.TclError, AttributeError):
            pass
        try:
            self.root.config(cursor="")
        except tk.TclError:
            pass
        
        # Verificar si fue cancelación
        if "Cancelado" in error_msg or self.cancel_event.is_set():
            self._update_status("Cancelado", "orange")
            self._log_message("✗ Ejecución cancelada por el usuario")
            messagebox.showinfo(
                "Cancelado",
                "La ejecución del pipeline fue cancelada.\n\n"
                "Los cambios realizados hasta el momento se descartaron.",
                parent=self.root
            )
        else:
            self._update_status("Error", "red")
            self._log_message(f"✗ ERROR: {error_msg}")
            messagebox.showerror("Error", f"Error durante la ejecución:\n\n{error_msg}", parent=self.root)


# Diagnóstico visual en MercadoPipelinePage (ref/backup + insumos clave)
_MERCADO_BACKUP_CHECKS: list[tuple[str, str | None, str]] = [
    ("Matrículas 2019-2024", "backup/matriculas", "6 archivos matriculados_{año}.xlsx"),
    ("Primer Curso 2019-2024", "backup/matriculas primer curso", "6 archivos primer_curso_{año}.xlsx"),
    ("Graduados 2019-2024", "backup/graduados", "6 archivos graduados_{año}.xlsx"),
    ("Inscritos SNIES", "backup/inscritos", "Archivos inscritos_{año}.xlsx"),
    ("OLE Indicadores", "backup", "ole_indicadores.csv o .xlsx"),
    ("Instituciones IES", "backup/ies", "Instituciones.xlsx"),
    ("Programas SNIES", None, "Programas.xlsx"),
    ("Referente Categorías", "backup", "Referente_Categorias.xlsx"),
]


def _check_backup_source(ref_dir: Path, ruta_rel: str | None, etiqueta: str) -> tuple[str, str]:
    """
    Valida disponibilidad de insumos del estudio de mercado (ref/backup/, ref/, outputs/).
    Retorna (estado, detalle) donde estado es 'ok', 'parcial' o 'falta'.
    Usa comparaciones exactas de etiqueta para evitar matches accidentales.
    """
    if etiqueta == "Programas SNIES":
        try:
            from etl.config import OUTPUTS_DIR

            ok = (OUTPUTS_DIR / "Programas.xlsx").exists()
        except Exception:
            ok = False
        return ("ok" if ok else "falta", "Programas.xlsx")

    carpeta = ref_dir / ruta_rel if ruta_rel else ref_dir

    if etiqueta == "OLE Indicadores":
        ok = (carpeta / "ole_indicadores.csv").exists() or (carpeta / "ole_indicadores.xlsx").exists()
        return ("ok" if ok else "falta", "ole_indicadores.csv/.xlsx")

    if etiqueta == "Referente Categorías":
        ok = (carpeta / "Referente_Categorias.xlsx").exists()
        return ("ok" if ok else "falta", "Referente_Categorias.xlsx")

    if etiqueta == "Instituciones IES":
        ok = (carpeta / "Instituciones.xlsx").exists()
        return ("ok" if ok else "falta", "Instituciones.xlsx")

    if etiqueta == "Inscritos SNIES":
        if not carpeta.exists():
            return ("falta", "Carpeta no encontrada")
        archivos = list(carpeta.glob("inscritos_*.xlsx")) + list(carpeta.glob("inscritos_*.xls"))
        n = len(archivos)
        if n >= 6:
            return ("ok", f"{n} archivos encontrados")
        if n > 0:
            return ("parcial", f"Solo {n}/6 archivos")
        return ("falta", "Sin archivos inscritos_*.xlsx")

    if not carpeta.exists():
        return ("falta", "Carpeta no encontrada")

    years_ok: list[int] = []
    years_missing: list[int] = []
    for y in range(2019, 2025):
        hits = list(carpeta.glob(f"*{y}*.xlsx")) + list(carpeta.glob(f"*{y}*.xls"))
        if hits:
            years_ok.append(y)
        else:
            years_missing.append(y)

    if len(years_ok) == 6:
        return ("ok", "6/6 años disponibles (2019-2024)")
    if years_ok:
        faltantes = ", ".join(str(y) for y in years_missing)
        return ("parcial", f"{len(years_ok)}/6 años — faltan: {faltantes}")
    return ("falta", "Sin archivos encontrados")


class MercadoPipelinePage(ttk.Frame):
    """Página dedicada al pipeline de estudio de mercado Colombia, con progreso en tiempo real."""

    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        self.root = parent.winfo_toplevel()
        self.is_running = False
        self.cancel_event = threading.Event()
        self.seg_cancel_event = threading.Event()
        self._setup_ui()
        self._check_checkpoints()

    def _on_resize(self, w: int, h: int) -> None:
        """Ajusta el área de log según altura disponible."""
        try:
            log_lines = max(6, min(16, (h - 400) // 20))
            self.messages_text.config(height=log_lines)
        except (tk.TclError, AttributeError):
            pass
        try:
            wrap = max(500, w - 80)
            if hasattr(self, "seg_desc_label"):
                self.seg_desc_label.configure(wraplength=wrap)
        except (tk.TclError, AttributeError):
            pass

    def _populate_diagnostico_content(self, diag_card: ttk.Frame, ref_dir: Path | None) -> None:
        """Añade filas de semáforo y botón refrescar (asume título + subtítulo ya empaquetados)."""
        for w in diag_card.winfo_children()[2:]:
            w.destroy()

        _color_ok = EAFIT["success"]
        _color_parcial = EAFIT["warning"]
        _color_falta = EAFIT["danger"]
        _bg_ok = EAFIT["success_light"]
        _bg_parcial = EAFIT["warning_light"]
        _bg_falta = EAFIT["danger_light"]
        _icon_ok = "✅"
        _icon_parcial = "⚠️"
        _icon_falta = "❌"
        detail_bg = EAFIT["card_bg"]

        for etiqueta, ruta_rel, _desc in _MERCADO_BACKUP_CHECKS:
            fila = ttk.Frame(diag_card, style="Card.TFrame")
            fila.pack(fill=tk.X, pady=(0, 4))

            if ref_dir is None:
                estado, detalle = "falta", "REF_DIR no disponible"
            else:
                estado, detalle = _check_backup_source(ref_dir, ruta_rel, etiqueta)

            if estado == "ok":
                icon, fg, bg = _icon_ok, _color_ok, _bg_ok
            elif estado == "parcial":
                icon, fg, bg = _icon_parcial, _color_parcial, _bg_parcial
            else:
                icon, fg, bg = _icon_falta, _color_falta, _bg_falta

            tk.Label(
                fila,
                text=f" {icon} {etiqueta} ",
                bg=bg,
                fg=fg,
                font=("Segoe UI", 9, "bold"),
                padx=6,
                pady=2,
            ).pack(side=tk.LEFT)

            tk.Label(
                fila,
                text=f" — {detalle}",
                bg=detail_bg,
                fg=EAFIT["text_muted"],
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT)

        self.btn_refresh_diag = ttk.Button(
            diag_card,
            text="↻  Refrescar diagnóstico",
            command=self._refresh_diagnostico,
            style="Secondary.TButton",
        )
        self.btn_refresh_diag.pack(anchor="w", pady=(8, 0))

    def _refresh_diagnostico(self) -> None:
        try:
            from etl.config import REF_DIR as _ref

            self._diag_ref_dir = _ref
        except Exception:
            self._diag_ref_dir = None
        self._populate_diagnostico_content(self._diag_card, self._diag_ref_dir)

    def _setup_ui(self):
        # ── Contenedor raíz: header fijo arriba + área scrollable abajo ──────
        root_frame = ttk.Frame(self, style="Page.TFrame")
        root_frame.pack(fill=tk.BOTH, expand=True)

        # Header fijo (NO entra al scroll)
        header = ttk.Frame(root_frame, padding=(20, 14, 20, 0), style="Page.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="📊 Estudio de Mercado Colombia", style="Header.TLabel").pack(side=tk.LEFT)
        if self.on_back:
            self.btn_mercado_back = ttk.Button(
                header,
                text="← Volver al menú",
                command=lambda: self.on_back() if self.on_back else None,
                style="Back.TButton",
            )
            self.btn_mercado_back.pack(side=tk.RIGHT)

        # ── Canvas + scrollbar vertical ──────────────────────────────────────
        canvas_outer = ttk.Frame(root_frame, style="Page.TFrame")
        canvas_outer.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(
            canvas_outer, highlightthickness=0,
            bg=EAFIT["bg"],
        )
        vscroll = ttk.Scrollbar(canvas_outer, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame interno que contiene todos los cards
        main_frame = ttk.Frame(self._scroll_canvas, padding=(20, 10, 20, 20), style="Page.TFrame")
        self._canvas_window = self._scroll_canvas.create_window(
            (0, 0), window=main_frame, anchor="nw"
        )

        # Ajustar ancho del frame interno al canvas
        def _on_canvas_configure(event):
            self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)

        def _on_frame_configure(event):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
            # Mostrar/ocultar scrollbar según necesidad
            bbox = self._scroll_canvas.bbox("all")
            if bbox:
                content_h = bbox[3] - bbox[1]
                canvas_h  = self._scroll_canvas.winfo_height()
                if content_h > canvas_h and canvas_h > 1:
                    vscroll.pack(side=tk.RIGHT, fill=tk.Y)
                else:
                    vscroll.pack_forget()

        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)
        main_frame.bind("<Configure>", _on_frame_configure)

        # Mousewheel
        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Card: Diagnóstico de datos disponibles ───────────────────────────
        diag_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        diag_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(
            diag_card,
            text="🔍 Diagnóstico de datos disponibles",
            style="SectionTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            diag_card,
            text="Estado de los archivos en ref/backup/ antes de ejecutar el pipeline.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 10))
        try:
            from etl.config import REF_DIR as _ref_dir_diag, OUTPUTS_DIR as _outputs_dir_diag
        except Exception:
            _ref_dir_diag = None
            _outputs_dir_diag = None
        self._diag_card = diag_card
        self._diag_ref_dir = _ref_dir_diag
        self._populate_diagnostico_content(diag_card, _ref_dir_diag)

        # ── Card: Checkpoints ────────────────────────────────────────────────
        cp_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        cp_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(cp_card, text="⚡ Checkpoints disponibles", style="SectionTitle.TLabel").pack(anchor="w")
        self.reuse_base_var = tk.BooleanVar(value=False)
        self.reuse_sabana_var = tk.BooleanVar(value=False)
        self.cb_base = ttk.Checkbutton(
            cp_card,
            text="Reusar base_maestra.parquet (omitir Fase 1 — clasificación ML)",
            variable=self.reuse_base_var,
            state=tk.DISABLED,
        )
        self.cb_base.pack(anchor="w", pady=(8, 0))
        ttk.Label(cp_card, text="Actívalo si ya ejecutaste la fase anteriormente y no hay cambios.", style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        self.cb_sabana = ttk.Checkbutton(
            cp_card,
            text="Reusar sabana_consolidada.parquet (omitir Fase 3 — consolidación)",
            variable=self.reuse_sabana_var,
            state=tk.DISABLED,
        )
        self.cb_sabana.pack(anchor="w", pady=(0, 0))
        ttk.Label(cp_card, text="Actívalo si ya ejecutaste la fase anteriormente y no hay cambios.", style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        # ── Card: SMLMV vigente ──────────────────────────────────────────────
        smlmv_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        smlmv_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(smlmv_card, text="💰 SMLMV vigente", style="SectionTitle.TLabel").pack(anchor="w")
        current_smlmv = get_smlmv_sesion()
        formatted_smlmv = f"{current_smlmv:,.0f}".replace(",", ".")
        self.smlmv_label = ttk.Label(smlmv_card, text=f"Valor actual: ${formatted_smlmv}", style="Muted.TLabel")
        self.smlmv_label.pack(anchor="w", pady=(4, 4))
        entry_row = ttk.Frame(smlmv_card, style="Card.TFrame")
        entry_row.pack(fill=tk.X, pady=(0, 0))
        ttk.Label(entry_row, text="Nuevo valor:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.smlmv_var = tk.StringVar(value=str(int(current_smlmv)))
        vcmd = (self.register(self._validate_digits), "%P")
        self.smlmv_entry = ttk.Entry(entry_row, textvariable=self.smlmv_var, width=12, validate="key", validatecommand=vcmd)
        self.smlmv_entry.pack(side=tk.LEFT, padx=(6, 6))
        self.btn_smlmv_update = ttk.Button(
            entry_row, text="Actualizar", command=self._update_smlmv, style="Secondary.TButton"
        )
        self.btn_smlmv_update.pack(side=tk.LEFT)
        ttk.Label(
            smlmv_card,
            text="El valor se usa en la Fase 4 para calcular salarios en SMLMV. Se guarda en config.json y persiste entre sesiones.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        # ── Card: Benchmarks de costo por nivel ─────────────────────────────
        try:
            from etl.config import get_todos_benchmarks
            _bench_available = True
        except ImportError:
            _bench_available = False

        if _bench_available:
            self._mercado_bench_widgets: list[tk.Misc] = []
            bench_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
            bench_card.pack(fill=tk.X, pady=(0, 14))
            ttk.Label(bench_card, text="🏷️ Benchmarks de costo de matrícula por nivel", style="SectionTitle.TLabel").pack(anchor="w")
            ttk.Label(
                bench_card,
                text="Precios de referencia para comparar el costo promedio de cada categoría según nivel. Se guardan en config.json.",
                style="Muted.TLabel",
            ).pack(anchor="w", pady=(4, 10), fill=tk.X)

            vcmd_bench = (self.register(self._validate_digits), "%P")
            benchmarks_actuales = get_todos_benchmarks()

            # Filas: (label_texto, nivel_key, atributo_var, atributo_label)
            _bench_niveles = [
                ("Pregrado / Tecnológico ($):", "pregrado",        "bench_var_pre",  "bench_lbl_pre"),
                ("Especialización ($):",        "especializacion",  "bench_var_esp",  "bench_lbl_esp"),
                ("Maestría ($):",               "maestria",         "bench_var_mae",  "bench_lbl_mae"),
                ("Doctorado ($):",              "doctorado",        "bench_var_doc",  "bench_lbl_doc"),
            ]

            self._bench_vars = {}
            self._bench_labels = {}

            for label_txt, nivel_key, var_attr, lbl_attr in _bench_niveles:
                fila = ttk.Frame(bench_card, style="Card.TFrame")
                fila.pack(fill=tk.X, pady=(0, 6))

                ttk.Label(fila, text=label_txt, style="Muted.TLabel", width=26, anchor="w").pack(side=tk.LEFT)

                valor_actual = benchmarks_actuales.get(nivel_key, 13_400_000)
                formatted = f"{valor_actual:,.0f}".replace(",", ".")
                lbl = ttk.Label(fila, text=f"${formatted}", style="Muted.TLabel", width=16, anchor="w")
                lbl.pack(side=tk.LEFT, padx=(4, 8))
                self._bench_labels[nivel_key] = lbl

                var = tk.StringVar(value=str(int(valor_actual)))
                self._bench_vars[nivel_key] = var
                ttk.Entry(fila, textvariable=var, width=13, validate="key", validatecommand=vcmd_bench).pack(side=tk.LEFT)
                _ent_b = fila.winfo_children()[-1]
                self._mercado_bench_widgets.append(_ent_b)

                # Capturar nivel_key para el closure
                _btn_b = ttk.Button(
                    fila,
                    text="Actualizar",
                    command=lambda nk=nivel_key: self._update_benchmark(nk),
                    style="Secondary.TButton",
                )
                _btn_b.pack(side=tk.LEFT, padx=(6, 0))
                self._mercado_bench_widgets.append(_btn_b)
        else:
            self._mercado_bench_widgets = []

        # ── Card: Fase 1 — clasificación ─────────────────────────────────────
        fase1_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        fase1_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(fase1_card, text="1. Fase 1 — Clasificar programas por categoría", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(
            fase1_card,
            text="Cruza Programas.xlsx con el referente de categorías y clasifica cada programa usando cascada SNIES → Nombre → KNN TF-IDF. Genera un Excel descargable. El checkpoint queda guardado para las fases siguientes.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(6, 12), fill=tk.X)
        btn_frame_f1 = ttk.Frame(fase1_card, style="Card.TFrame")
        btn_frame_f1.pack(fill=tk.X)
        self.btn_fase1 = ttk.Button(
            btn_frame_f1,
            text="▶️ Ejecutar Fase 1 → Excel",
            command=self._on_fase1_clicked,
            style="Primary.TButton",
        )
        self.btn_fase1.pack(side=tk.LEFT)
        self.btn_cancel_fase1 = ttk.Button(
            btn_frame_f1,
            text="⏹️ Cancelar",
            command=self._on_cancel_clicked,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.btn_cancel_fase1.pack(side=tk.LEFT, padx=(10, 0))
        self.btn_ver_programas = ttk.Button(
            btn_frame_f1,
            text="📋 Ver Programas con Categorías",
            command=self._open_programas_categorias,
            style="Secondary.TButton",
        )
        self.btn_ver_programas.pack(side=tk.LEFT, padx=(10, 0))
        self.progress_fase1 = ttk.Progressbar(fase1_card, mode="indeterminate")
        self.progress_fase1.pack(fill=tk.X, pady=(12, 0))
        self.progress_label_fase1 = ttk.Label(fase1_card, text="", style="Status.TLabel")
        self.progress_label_fase1.pack(anchor="w", pady=(4, 0))

        # ── Card: Fases 2-5 — pipeline completo ─────────────────────────────
        run_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        run_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(run_card, text="2. Fases 2-5 — Pipeline completo → Estudio de Mercado", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(
            run_card,
            text="Consolida matrículas históricas (2019-2024), OLE, costos y scoring. Requiere haber ejecutado la Fase 1 primero. Genera Estudio_Mercado_Colombia.xlsx.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(6, 12), fill=tk.X)
        btn_frame = ttk.Frame(run_card, style="Card.TFrame")
        btn_frame.pack(fill=tk.X)
        self.btn_execute = ttk.Button(
            btn_frame,
            text="▶️ Ejecutar Fases 2-5",
            command=self._on_execute_clicked,
            style="Primary.TButton",
        )
        self.btn_execute.pack(side=tk.LEFT)
        self.btn_cancel = ttk.Button(btn_frame, text="⏹️ Cancelar", command=self._on_cancel_clicked, state=tk.DISABLED, style="Danger.TButton")
        self.btn_cancel.pack(side=tk.LEFT, padx=(10, 0))
        self.btn_resultado = ttk.Button(btn_frame, text="📂 Ver resultado", command=self._open_resultado, state=tk.DISABLED, style="Secondary.TButton")
        self.btn_resultado.pack(side=tk.LEFT, padx=(10, 0))
        self.lbl_checkpoint = ttk.Label(run_card, text="", style="Status.TLabel")
        self.lbl_checkpoint.pack(anchor="w", pady=(8, 0))
        self.progress = ttk.Progressbar(run_card, mode="determinate", maximum=4, value=0)
        self.progress.pack(fill=tk.X, pady=(12, 0))
        self.progress_label = ttk.Label(run_card, text="Progreso: listo", style="Status.TLabel")
        self.progress_label.pack(anchor="w", pady=(4, 0))

        # ── Card 3: Segmentos ────────────────────────────────────────────
        seg_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        seg_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(
            seg_card,
            text="3. Reportes segmentados — Bogotá · Antioquia · Eje Cafetero · Virtual",
            style="SectionTitle.TLabel",
        ).pack(anchor="w")
        self.seg_desc_label = ttk.Label(
            seg_card,
            text=(
                "Recalcula scoring, AAGR y semáforos de calidad de forma independiente "
                "para cada segmento. Requiere haber ejecutado las Fases 2-5 primero."
            ),
            style="Muted.TLabel",
            wraplength=900,
        )
        self.seg_desc_label.pack(anchor="w", pady=(4, 10))
        self.var_force_recalc = tk.BooleanVar(value=False)
        self.cb_force_seg = ttk.Checkbutton(
            seg_card,
            text="Forzar recálculo completo (ignorar caché)",
            variable=self.var_force_recalc,
        )
        self.cb_force_seg.pack(anchor="w", pady=(0, 6))
        btn_frame_seg = ttk.Frame(seg_card, style="Card.TFrame")
        btn_frame_seg.pack(anchor="w")
        self.btn_segmentos = ttk.Button(
            btn_frame_seg,
            text="Generar Reportes Segmentados",
            style="Primary.TButton",
            command=self._on_segmentos_clicked,
        )
        self.btn_segmentos.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_cancel_seg = ttk.Button(
            btn_frame_seg,
            text="Cancelar",
            style="Secondary.TButton",
            state=tk.DISABLED,
            command=self._on_cancel_segmentos_clicked,
        )
        self.btn_cancel_seg.pack(side=tk.LEFT)
        self.prog_seg = ttk.Progressbar(seg_card, mode="indeterminate", length=500)
        self.prog_seg.pack(fill=tk.X, pady=(10, 4))

        # ── Card 4: Valorización de programas ────────────────────────────────
        val_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        val_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(
            val_card,
            text="4. Valorización de Programas EAFIT",
            style="SectionTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            val_card,
            text=(
                "Genera Programas_para_valorizacion_output.xlsx con dos matrices "
                "por programa y región: métricas del mercado regional (todas las IES) "
                "y métricas de las 13 IES referentes. "
                "Requiere haber ejecutado los reportes segmentados primero."
            ),
            style="Muted.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(4, 10), fill=tk.X)
        btn_frame_val = ttk.Frame(val_card, style="Card.TFrame")
        btn_frame_val.pack(anchor="w")
        self.btn_valorizacion = ttk.Button(
            btn_frame_val,
            text="▶️ Generar Valorización",
            command=self._on_valorizacion_clicked,
            style="Primary.TButton",
        )
        self.btn_valorizacion.pack(side=tk.LEFT)
        self.btn_ver_valorizacion = ttk.Button(
            btn_frame_val,
            text="📂 Ver resultado",
            command=self._open_resultado_valorizacion,
            state=tk.DISABLED,
            style="Secondary.TButton",
        )
        self.btn_ver_valorizacion.pack(side=tk.LEFT, padx=(10, 0))
        self.prog_val = ttk.Progressbar(val_card, mode="indeterminate", length=500)
        self.prog_val.pack(fill=tk.X, pady=(10, 4))
        self.lbl_val_status = ttk.Label(val_card, text="", style="Status.TLabel")
        self.lbl_val_status.pack(anchor="w")

        # ── Card: Logs ───────────────────────────────────────────────────────
        log_card = ttk.Frame(main_frame, padding=16, style="Card.TFrame")
        log_card.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(log_card, text="📋 Progreso detallado", style="SectionTitle.TLabel").pack(anchor="w")
        scrollbar = ttk.Scrollbar(log_card)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.messages_text = tk.Text(
            log_card,
            height=10,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            font=("Consolas", 9),
            bg=EAFIT["card_bg"],
            fg=EAFIT["text"],
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.messages_text.yview)

        self._mercado_init_busy_helpers()

    def _mercado_init_busy_helpers(self) -> None:
        if not getattr(self, "_mercado_bench_widgets", None):
            self._mercado_bench_widgets = []
        self._mercado_secondary_widgets = [
            w
            for w in (
                getattr(self, "btn_mercado_back", None),
                getattr(self, "btn_refresh_diag", None),
                self.cb_base,
                self.cb_sabana,
                self.smlmv_entry,
                getattr(self, "btn_smlmv_update", None),
                self.btn_fase1,
                self.btn_ver_programas,
                self.btn_execute,
                self.btn_resultado,
                self.btn_segmentos,
                getattr(self, "cb_force_seg", None),
                self.btn_valorizacion,
                self.btn_ver_valorizacion,
            )
            if w is not None
        ]
        self._mercado_bench_busy = list(self._mercado_bench_widgets)

    def _mercado_refresh_resultado_button_state(self) -> None:
        try:
            from etl.config import ARCHIVO_ESTUDIO_MERCADO

            self.btn_resultado.config(
                state=tk.NORMAL if ARCHIVO_ESTUDIO_MERCADO.exists() else tk.DISABLED
            )
        except (tk.TclError, AttributeError):
            pass

    def _mercado_set_secondary_busy(self, busy: bool) -> None:
        """Deshabilita controles secundarios (y benchmarks) durante hilos largos; no toca los botones Cancelar."""
        st = tk.DISABLED if busy else tk.NORMAL
        for w in getattr(self, "_mercado_secondary_widgets", ()):
            try:
                w.config(state=st)
            except (tk.TclError, AttributeError):
                pass
        for w in getattr(self, "_mercado_bench_busy", ()):
            try:
                w.config(state=st)
            except (tk.TclError, AttributeError):
                pass
        try:
            self.root.config(cursor="watch" if busy else "")
        except (tk.TclError, AttributeError):
            pass
        if not busy:
            self._check_checkpoints(quiet=True)
            self._mercado_refresh_resultado_button_state()

    def _on_valorizacion_clicked(self):
        """Ejecuta la Fase 7 — Valorización de Programas."""
        import threading
        from etl.valorizacion_pipeline import run_fase_valorizacion

        if self.is_running:
            messagebox.showwarning(
                "Atención",
                "Hay un proceso del estudio de mercado en curso. Espera a que termine o cancélalo.",
                parent=self.root,
            )
            return

        self._mercado_set_secondary_busy(True)
        self.prog_val.start(12)
        self.lbl_val_status.config(text="⏳ Generando valorización...")
        self._log_message("━━━ Fase 7 — Valorización de Programas ━━━")

        def _run():
            try:
                ruta = run_fase_valorizacion(log=self._log_message)
                self.after(0, lambda p=ruta: self._on_valorizacion_done(p))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_valorizacion_error(e))

        threading.Thread(target=_run, daemon=True).start()

    def _on_valorizacion_done(self, ruta: Path):
        self.prog_val.stop()
        self._mercado_set_secondary_busy(False)
        self.btn_valorizacion.config(state=tk.NORMAL)
        self.btn_ver_valorizacion.config(state=tk.NORMAL)
        self._val_ruta = ruta
        self.lbl_val_status.config(text=f"✅ Generado: {ruta.name}")
        self._log_message(f"✓ Valorización completada: {ruta}")

    def _on_valorizacion_error(self, exc: Exception):
        self.prog_val.stop()
        self._mercado_set_secondary_busy(False)
        self.btn_valorizacion.config(state=tk.NORMAL)
        self.btn_ver_valorizacion.config(state=tk.DISABLED)
        self.lbl_val_status.config(text=f"❌ Error: {exc}")
        self._log_message(f"✗ Error en valorización: {exc}")
        messagebox.showerror(
            "Error en Valorización",
            f"No se pudo generar el archivo:\n\n{exc}",
            parent=self.root,
        )

    def _open_resultado_valorizacion(self):
        ruta = getattr(self, "_val_ruta", None)
        if ruta and ruta.exists():
            _open_in_excel(ruta)
        else:
            from etl.config import ESTUDIO_MERCADO_DIR

            candidato = ESTUDIO_MERCADO_DIR / "Programas_para_valorizacion_output.xlsx"
            if candidato.exists():
                _open_in_excel(candidato)
            else:
                messagebox.showinfo(
                    "Sin resultado",
                    "Aún no se ha generado el archivo de valorización.\n"
                    "Ejecuta primero 'Generar Valorización'.",
                    parent=self.root,
                )

    def _open_programas_categorias(self):
        """Abre una ventana con Programas.xlsx mostrando CATEGORIA_FINAL y fuente ML."""
        from etl.config import ARCHIVO_PROGRAMAS
        from etl.exceptions_helpers import leer_excel_con_reintentos

        if not ARCHIVO_PROGRAMAS.exists():
            safe_messagebox_error(
                "Sin datos",
                "No existe outputs/Programas.xlsx.\nEjecuta primero el pipeline principal (análisis SNIES).",
                parent=self.root,
            )
            return

        try:
            df = leer_excel_con_reintentos(ARCHIVO_PROGRAMAS, sheet_name="Programas")
        except Exception as exc:
            safe_messagebox_error("Error", f"No se pudo leer Programas.xlsx:\n{exc}", parent=self.root)
            return

        # Ventana emergente
        win = tk.Toplevel(self.root)
        win.title("Programas con Categorías de Mercado")
        win.geometry("1200x650")
        win.minsize(800, 400)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = ttk.Frame(win, padding=(12, 10), style="Page.TFrame")
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="📋 Programas con Categorías de Mercado", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            hdr,
            text=f"{len(df):,} programas  |  {ARCHIVO_PROGRAMAS.name}",
            foreground=EAFIT["text_muted"],
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=(16, 0))

        # ── Filtros ──────────────────────────────────────────────────────────
        flt = ttk.Frame(win, padding=(12, 6), style="Page.TFrame")
        flt.pack(fill=tk.X)

        ttk.Label(flt, text="Fuente:").pack(side=tk.LEFT, padx=(0, 6))
        fuente_var = tk.StringVar(value="TODAS")
        fuentes_disponibles = ["TODAS"]
        if "FUENTE_CATEGORIA" in df.columns:
            fuentes_disponibles += sorted(df["FUENTE_CATEGORIA"].dropna().astype(str).unique().tolist())
        ttk.Combobox(flt, textvariable=fuente_var, values=fuentes_disponibles, state="readonly", width=16).pack(side=tk.LEFT)

        ttk.Label(flt, text="Nivel:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 6))
        nivel_var = tk.StringVar(value="TODOS")
        niveles_disp = ["TODOS"]
        if "NIVEL_DE_FORMACIÓN" in df.columns:
            niveles_disp += sorted(df["NIVEL_DE_FORMACIÓN"].dropna().astype(str).unique().tolist())
        ttk.Combobox(flt, textvariable=nivel_var, values=niveles_disp, state="readonly", width=22).pack(side=tk.LEFT)

        ttk.Label(flt, text="Buscar:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 6))
        buscar_var = tk.StringVar()
        ttk.Entry(flt, textvariable=buscar_var, width=26).pack(side=tk.LEFT)

        page_label = ttk.Label(flt, text="", foreground=EAFIT["text_muted"])
        page_label.pack(side=tk.RIGHT)
        ttk.Button(flt, text="Siguiente ›", command=lambda: cambiar_pagina(1)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(flt, text="‹ Anterior", command=lambda: cambiar_pagina(-1)).pack(side=tk.RIGHT, padx=4)

        # ── Tabla ───────────────────────────────────────────────────────────
        COLS = [
            "CÓDIGO_SNIES_DEL_PROGRAMA", "NOMBRE_DEL_PROGRAMA", "NOMBRE_INSTITUCIÓN",
            "NIVEL_DE_FORMACIÓN", "CATEGORIA_FINAL", "FUENTE_CATEGORIA",
            "PROBABILIDAD", "REQUIERE_REVISION", "ESTADO_PROGRAMA",
            "DEPARTAMENTO_OFERTA_PROGRAMA",
        ]
        cols_show = [c for c in COLS if c in df.columns]

        tbl_frame = ttk.Frame(win)
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 4))
        tree = ttk.Treeview(tbl_frame, columns=cols_show, show="headings", height=22)
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.grid_rowconfigure(0, weight=1)
        tbl_frame.grid_columnconfigure(0, weight=1)

        anchos = {
            "CÓDIGO_SNIES_DEL_PROGRAMA": 120, "NOMBRE_DEL_PROGRAMA": 280,
            "NOMBRE_INSTITUCIÓN": 200, "NIVEL_DE_FORMACIÓN": 160,
            "CATEGORIA_FINAL": 200, "FUENTE_CATEGORIA": 120,
            "PROBABILIDAD": 90, "REQUIERE_REVISION": 110,
            "ESTADO_PROGRAMA": 90, "DEPARTAMENTO_OFERTA_PROGRAMA": 160,
        }
        for c in cols_show:
            tree.heading(c, text=c)
            tree.column(c, width=anchos.get(c, 130), minwidth=80, anchor="w")

        # Colores por fuente
        tree.tag_configure("CRUCE_SNIES",  background="#C6EFCE")
        tree.tag_configure("MATCH_NOMBRE", background="#E2EFDA")
        tree.tag_configure("KNN_TFIDF",    background="#FFEB9C")
        tree.tag_configure("REQUIERE_REV", background="#FFC7CE")

        PAGE_SIZE = 200
        state = {"page": 0, "df": df}

        def _renderizar(df_filtrado):
            for item in tree.get_children():
                tree.delete(item)
            total = len(df_filtrado)
            max_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            p = max(0, min(state["page"], max_pages - 1))
            state["page"] = p
            start = p * PAGE_SIZE
            end = min(total, start + PAGE_SIZE)
            page_label.config(text=f"Página {p+1}/{max_pages}  ({total:,} filas)")
            for _, row in df_filtrado.iloc[start:end].iterrows():
                vals = [str(row.get(c, "") or "") for c in cols_show]
                fuente = str(row.get("FUENTE_CATEGORIA", "")).upper().strip()
                req = str(row.get("REQUIERE_REVISION", "")).lower() in ("true", "1", "yes", "sí")
                tag = "REQUIERE_REV" if req else fuente if fuente in ("CRUCE_SNIES", "MATCH_NOMBRE", "KNN_TFIDF") else ""
                tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

        def aplicar():
            df_f = df.copy()
            fuente_sel = fuente_var.get()
            if fuente_sel != "TODAS" and "FUENTE_CATEGORIA" in df_f.columns:
                df_f = df_f[df_f["FUENTE_CATEGORIA"].astype(str).str.upper().str.strip() == fuente_sel.upper()]
            nivel_sel = nivel_var.get()
            if nivel_sel != "TODOS" and "NIVEL_DE_FORMACIÓN" in df_f.columns:
                df_f = df_f[df_f["NIVEL_DE_FORMACIÓN"].astype(str).str.upper().str.strip() == nivel_sel.upper()]
            q = buscar_var.get().strip().lower()
            if q:
                mask = pd.Series(False, index=df_f.index)
                for col in ("NOMBRE_DEL_PROGRAMA", "CATEGORIA_FINAL", "NOMBRE_INSTITUCIÓN"):
                    if col in df_f.columns:
                        mask |= df_f[col].astype(str).str.lower().str.contains(q, na=False)
                df_f = df_f[mask]
            state["df"] = df_f
            state["page"] = 0
            _renderizar(df_f)

        def cambiar_pagina(delta):
            state["page"] += delta
            _renderizar(state["df"])

        # Bindear filtros
        fuente_var.trace_add("write", lambda *_: aplicar())
        nivel_var.trace_add("write", lambda *_: aplicar())
        buscar_var.trace_add("write", lambda *_: aplicar())

        # Renderizado inicial
        aplicar()
        win.grab_set()

    def _on_segmentos_clicked(self) -> None:
        from etl.config import CHECKPOINT_BASE_MAESTRA

        sabana_path = CHECKPOINT_BASE_MAESTRA.parent / "sabana_consolidada.parquet"
        ag_path = CHECKPOINT_BASE_MAESTRA.parent / "agregado_categorias.parquet"

        if not sabana_path.exists() or not ag_path.exists():
            messagebox.showwarning(
                "Fases incompletas",
                "Ejecuta primero las Fases 1-5 (nacional) antes de generar los reportes segmentados.",
                parent=self.root,
            )
            return
        if self.is_running:
            messagebox.showwarning(
                "Atención",
                "Espera a que termine el pipeline en curso o cancélalo antes de generar reportes segmentados.",
                parent=self.root,
            )
            return

        self.seg_cancel_event.clear()
        self._mercado_set_secondary_busy(True)
        self.btn_cancel_seg.config(state=tk.NORMAL)
        self.prog_seg.start(12)
        self._log_message("Iniciando generación de reportes segmentados...")

        threading.Thread(target=self._run_segmentos_thread, daemon=True).start()

    def _on_cancel_segmentos_clicked(self) -> None:
        self.seg_cancel_event.set()
        self._log_message("[Segmentos] Cancelación solicitada; se detendrá al terminar el segmento en curso.")

    def _run_segmentos_thread(self) -> None:
        from etl.config import CHECKPOINT_BASE_MAESTRA
        from etl.mercado_pipeline import run_segmentos_regionales

        try:
            base_dir = get_configured_base_dir()
            if not base_dir or not base_dir.exists():
                self.root.after(0, lambda: self._on_segmentos_error("No hay carpeta del proyecto configurada."))
                return
            update_paths_for_base_dir(base_dir)

            sabana_path = CHECKPOINT_BASE_MAESTRA.parent / "sabana_consolidada.parquet"
            ag_path = CHECKPOINT_BASE_MAESTRA.parent / "agregado_categorias.parquet"

            sabana = pd.read_parquet(sabana_path)
            ag_nac = pd.read_parquet(ag_path)

            resultados = run_segmentos_regionales(
                sabana,
                ag_nac,
                cancel_event=self.seg_cancel_event,
                force_recalc=self.var_force_recalc.get(),
            )

            self.root.after(0, self._on_segmentos_completed, resultados)
        except Exception as e:
            self.root.after(0, self._on_segmentos_error, str(e))

    def _on_segmentos_completed(self, resultados: dict) -> None:
        self.prog_seg.stop()
        self._mercado_set_secondary_busy(False)
        self.btn_segmentos.config(state=tk.NORMAL)
        self.btn_cancel_seg.config(state=tk.DISABLED)
        nombres = ", ".join(resultados.keys()) if resultados else "ninguno"
        self._log_message(f"Reportes segmentados listos: {nombres}")
        from etl.config import ESTUDIO_MERCADO_DIR

        messagebox.showinfo(
            "Reportes segmentados generados",
            f"Se generaron {len(resultados)} archivos Excel:\n\n"
            + "\n".join(f"  Estudio_Mercado_{k}.xlsx" for k in resultados)
            + f"\n\nUbicación: {ESTUDIO_MERCADO_DIR}",
            parent=self.root,
        )

    def _on_segmentos_error(self, error: str) -> None:
        self.prog_seg.stop()
        self._mercado_set_secondary_busy(False)
        self.btn_segmentos.config(state=tk.NORMAL)
        self.btn_cancel_seg.config(state=tk.DISABLED)
        self._log_message(f"Error en segmentos: {error}")
        messagebox.showerror("Error en segmentos", error, parent=self.root)

    def _check_checkpoints(self, quiet: bool = False):
        from etl.config import CHECKPOINT_BASE_MAESTRA
        sabana_path = CHECKPOINT_BASE_MAESTRA.parent / "sabana_consolidada.parquet"
        if CHECKPOINT_BASE_MAESTRA.exists():
            self.cb_base.config(state=tk.NORMAL)
            self.reuse_base_var.set(True)
            if not quiet:
                self._log_message("✓ Checkpoint base_maestra encontrado")
        if sabana_path.exists():
            self.cb_sabana.config(state=tk.NORMAL)
            self.reuse_sabana_var.set(True)
            if not quiet:
                self._log_message("✓ Checkpoint sabana_consolidada encontrado")
        if not CHECKPOINT_BASE_MAESTRA.exists() and not sabana_path.exists():
            if not quiet:
                self._log_message("Sin checkpoints. Se ejecutarán todas las fases.")
        # Mostrar el SMLMV efectivo que se usará en el scoring
        if not quiet:
            try:
                smlmv = get_smlmv_sesion()
                formatted = f"{smlmv:,.0f}".replace(",", ".")
                self._log_message(f"💰 SMLMV vigente: ${formatted}. Puedes ajustarlo antes de ejecutar.")
            except Exception:
                pass
        self.root.after(200, self._refresh_checkpoint_label)

    def _validate_digits(self, value: str) -> bool:
        """Validador simple para permitir solo dígitos (o vacío) en el Entry de SMLMV."""
        return value.isdigit() or value == ""

    def _update_smlmv(self):
        """Actualiza el SMLMV de sesión desde el Entry y refresca la etiqueta."""
        raw = (self.smlmv_var.get() or "").strip()
        if not raw.isdigit():
            safe_messagebox_error("Valor inválido", "Ingresa un número entero para el SMLMV.", parent=self.root)
            return
        try:
            valor = int(raw)
            if valor <= 0:
                raise ValueError("El SMLMV debe ser positivo.")
        except Exception as e:
            safe_messagebox_error("Valor inválido", str(e), parent=self.root)
            return
        set_smlmv_sesion(valor)
        formatted = f"{valor:,.0f}".replace(",", ".")
        self.smlmv_label.config(text=f"Valor actual: ${formatted}")
        self._log_message(f"SMLMV de sesión actualizado a ${formatted}")

    def _update_benchmark(self, nivel: str = "general"):
        if not getattr(self, "_bench_vars", None):
            return
        var = self._bench_vars.get(nivel)
        if var is None:
            return
        raw = (var.get() or "").strip()
        if not raw.isdigit():
            safe_messagebox_error("Valor inválido", "Ingresa un número entero.", parent=self.root)
            return
        try:
            valor = int(raw)
            if valor <= 0:
                raise ValueError("El benchmark debe ser positivo.")
        except Exception as e:
            safe_messagebox_error("Valor inválido", str(e), parent=self.root)
            return
        ok = set_benchmark_costo(float(valor), nivel=nivel)
        if ok:
            formatted = f"{valor:,.0f}".replace(",", ".")
            lbl = self._bench_labels.get(nivel)
            if lbl:
                lbl.config(text=f"${formatted}")
            nombres = {"pregrado": "Pregrado", "especializacion": "Especialización",
                       "maestria": "Maestría", "doctorado": "Doctorado"}
            self._log_message(f"Benchmark {nombres.get(nivel, nivel)} actualizado a ${formatted}")
        else:
            safe_messagebox_error("Error", "No se pudo guardar en config.json.", parent=self.root)

    def _log_message(self, message: str):
        """Agrega un mensaje al área de texto (thread-safe vía root.after si se llama desde otro hilo)."""
        def _do():
            self.messages_text.config(state=tk.NORMAL)
            timestamp = time.strftime("%H:%M:%S")
            self.messages_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.messages_text.see(tk.END)
            self.messages_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
        try:
            self.root.after(0, _do)
        except Exception:
            _do()

    def _on_execute_clicked(self):
        if self.is_running:
            messagebox.showwarning("Atención", "Ya hay un proceso en ejecución.", parent=self.root)
            return
        from etl.config import CHECKPOINT_BASE_MAESTRA
        if not CHECKPOINT_BASE_MAESTRA.exists():
            messagebox.showerror(
                "Fase 1 requerida",
                "No se encontró el checkpoint de la Fase 1 (base_maestra.parquet).\n\n"
                "Ejecuta primero 'Ejecutar Fase 1 → Excel' y luego vuelve aquí.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Confirmar Fases 2-5",
            "¿Ejecutar las Fases 2-5 del estudio de mercado?\n\n"
            "Consolida matrículas, OLE, scoring y genera Estudio_Mercado_Colombia.xlsx.\n"
            "Puede tardar varios minutos.",
            parent=self.root,
        ):
            return
        self._execute_pipeline()

    def _on_cancel_clicked(self):
        if not self.is_running:
            return
        if not messagebox.askyesno(
            "Cancelar ejecución",
            "¿Está seguro de que desea cancelar?\n\nLos cambios realizados hasta el momento se perderán.",
            parent=self.root,
        ):
            return
        self.cancel_event.set()
        # Feedback inmediato en la UI: la cancelación se aplica al finalizar la fase actual
        self.btn_cancel.config(state=tk.DISABLED)
        try:
            self.btn_cancel_fase1.config(state=tk.DISABLED)
        except Exception:
            pass
        try:
            self.progress_label.config(
                text="Progreso: cancelando (se detendrá al final de la fase actual)",
                foreground=EAFIT["warning"],
            )
        except Exception:
            pass
        try:
            self.progress_label_fase1.config(
                text="Cancelando (se detendrá al final de la fase actual)",
                foreground=EAFIT["warning"],
            )
        except Exception:
            pass
        self._log_message("[CANCELADO] Cancelación solicitada por el usuario. La ejecución se detendrá al finalizar la fase en curso.")

    def _open_resultado(self):
        if not ensure_base_dir(self.root, prompt_if_missing=False):
            return
        from etl.config import ARCHIVO_ESTUDIO_MERCADO
        if not ARCHIVO_ESTUDIO_MERCADO.exists():
            safe_messagebox_error(
                "Sin resultado",
                "Ejecuta el pipeline primero para generar el archivo.",
                parent=self.root,
            )
            return
        root = self.winfo_toplevel()
        if hasattr(root, "_main_menu_gui"):
            root._main_menu_gui._show_page("mercado_results", EstudioMercadoResultsPage)

    def _execute_pipeline(self):
        # Persistir el SMLMV actual antes de lanzar el pipeline
        try:
            raw = (self.smlmv_var.get() or "").strip()
            if raw.isdigit():
                set_smlmv_sesion(float(raw))
        except Exception:
            # No bloquear la ejecución si falla la persistencia; se usará el último valor válido
            pass

        self.is_running = True
        self.cancel_event.clear()
        self._mercado_set_secondary_busy(True)
        self.btn_cancel.config(state=tk.NORMAL)
        self.progress.config(value=0)
        self.progress_label.config(text="Progreso: listo", foreground=EAFIT["text_muted"])
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete("1.0", tk.END)
        self.messages_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _update_progress(self, value: int, text: str):
        self.progress.config(value=value)
        self.progress_label.config(text=f"Progreso: {text}", foreground=EAFIT["text_muted"])

    def _release_ui_after_run(self):
        """Libera la UI tras la ejecución (siempre llamado desde finally). Detiene la barra y habilita Ejecutar."""
        try:
            self.progress.stop()
        except Exception:
            pass
        self.btn_cancel.config(state=tk.DISABLED)
        self.is_running = False
        self._mercado_set_secondary_busy(False)
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_fase1.config(state=tk.NORMAL)
        self._refresh_checkpoint_label()

    def _run_thread(self):
        try:
            base_dir = get_configured_base_dir()
            if not base_dir or not base_dir.exists():
                self.root.after(0, lambda: self._on_mercado_error("No hay carpeta del proyecto configurada."))
                return
            update_paths_for_base_dir(base_dir)
            reuse_sabana = self.reuse_sabana_var.get()

            from etl.config import CHECKPOINT_BASE_MAESTRA
            from etl.mercado_pipeline import run_fase2, run_fase3, run_fase4, run_fase5
            sabana_path = CHECKPOINT_BASE_MAESTRA.parent / "sabana_consolidada.parquet"

            # Fase 2
            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_mercado_error("Cancelado"))
                return
            self.root.after(0, lambda: self._log_message("Ejecutando Fase 2 (scrapers)..."))
            self.root.after(0, lambda: self._update_progress(1, "Fase 2..."))
            run_fase2()
            self.root.after(0, lambda: self._update_progress(1, "Fase 2 ✓"))

            # Fase 3
            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_mercado_error("Cancelado"))
                return
            if not sabana_path.exists() or not reuse_sabana:
                self.root.after(0, lambda: self._log_message("Ejecutando Fase 3 (consolidación)..."))
                self.root.after(0, lambda: self._update_progress(2, "Fase 3..."))
                run_fase3()
            else:
                self.root.after(0, lambda: self._log_message("Reusando sabana_consolidada.parquet"))
            self.root.after(0, lambda: self._update_progress(2, "Fase 3 ✓"))

            # Fase 4
            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_mercado_error("Cancelado"))
                return
            self.root.after(0, lambda: self._log_message("Ejecutando Fase 4 (agregación + scoring)..."))
            self.root.after(0, lambda: self._update_progress(3, "Fase 4..."))
            ag_pos, ag_pre = run_fase4()
            ag = ag_pos  # alias backward-compat para downstream que asume hoja `total`
            self.root.after(0, lambda: self._update_progress(3, "Fase 4 ✓"))

            # Fase 5
            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_mercado_error("Cancelado"))
                return
            self.root.after(0, lambda: self._log_message("Ejecutando Fase 5 (exportación)..."))
            self.root.after(0, lambda: self._update_progress(4, "Fase 5..."))
            run_fase5(ag_pos, ag_pre)
            self.root.after(0, lambda: self._update_progress(4, "Fase 5 ✓"))

            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_mercado_error("Cancelado"))
                return

            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_mercado_error("Cancelado"))
                return
            self.root.after(0, self._on_mercado_completed)
        except Exception as e:
            # Capturar el mensaje en una variable local para evitar problemas de alcance con lambdas diferidas
            msg = str(e)
            self.root.after(0, lambda msg=msg: self._on_mercado_error(msg))
        finally:
            self.root.after(0, self._release_ui_after_run)

    def _on_mercado_completed(self):
        self.is_running = False
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.btn_resultado.config(state=tk.NORMAL)
        self.progress_label.config(text="Progreso: completado", foreground=EAFIT["success"])
        self._log_message("=" * 50)
        self._log_message("✓ Pipeline de estudio de mercado completado")
        from etl.config import ARCHIVO_ESTUDIO_MERCADO, ESTUDIO_MERCADO_DIR
        messagebox.showinfo(
            "Éxito",
            f"Exportación guardada en:\n{ARCHIVO_ESTUDIO_MERCADO}\n\nCarpeta: {ESTUDIO_MERCADO_DIR}",
            parent=self.root,
        )

    def _on_mercado_error(self, error_msg: str):
        self.is_running = False
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        if "Cancelado" in error_msg or self.cancel_event.is_set():
            self.progress_label.config(text="Progreso: cancelado", foreground=EAFIT["warning"])
            self._log_message("✗ Ejecución cancelada por el usuario")
            messagebox.showinfo("Cancelado", "La ejecución fue cancelada.", parent=self.root)
        else:
            self.progress_label.config(text="Progreso: error", foreground=EAFIT["danger"])
            self._log_message(f"✗ ERROR: {error_msg}")
            messagebox.showerror("Error", f"Error durante la ejecución:\n\n{error_msg}", parent=self.root)

    def _refresh_checkpoint_label(self):
        try:
            from etl.config import CHECKPOINT_BASE_MAESTRA
            if CHECKPOINT_BASE_MAESTRA.exists():
                ts = time.strftime("%d/%m/%Y %H:%M", time.localtime(CHECKPOINT_BASE_MAESTRA.stat().st_mtime))
                self.lbl_checkpoint.config(text=f"✅ Fase 1 lista ({ts})", foreground=EAFIT["success"])
            else:
                self.lbl_checkpoint.config(text="⚠️ Ejecuta primero la Fase 1", foreground=EAFIT["warning"])
        except Exception:
            pass

    def _on_fase1_clicked(self):
        if self.is_running:
            messagebox.showwarning("Atención", "Ya hay un proceso en ejecución.", parent=self.root)
            return
        if not messagebox.askyesno(
            "Confirmar Fase 1",
            "¿Ejecutar la Fase 1 (clasificación de programas por categoría)?\n\n"
            "Al finalizar se pedirá dónde guardar el Excel con los resultados.\n"
            "Esto puede tardar varios minutos.",
            parent=self.root,
        ):
            return
        self._execute_fase1()

    def _execute_fase1(self):
        self.is_running = True
        self.cancel_event.clear()
        self._mercado_set_secondary_busy(True)
        self.btn_cancel_fase1.config(state=tk.NORMAL)
        self.lbl_checkpoint.config(text="Ejecutando Fase 1...", foreground=EAFIT["text_muted"])
        self.progress_fase1.start(12)
        self.progress_label_fase1.config(text="Ejecutando...", foreground=EAFIT["text_muted"])
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete("1.0", tk.END)
        self.messages_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_fase1_only_thread, daemon=True).start()

    def _run_fase1_only_thread(self):
        try:
            base_dir = get_configured_base_dir()
            if not base_dir or not base_dir.exists():
                self.root.after(0, lambda: self._on_fase1_error("No hay carpeta del proyecto configurada."))
                return
            update_paths_for_base_dir(base_dir)
            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_fase1_error("Cancelado"))
                return
            self.root.after(0, lambda: self._log_message("Iniciando Fase 1 — clasificación de programas..."))
            from etl.mercado_pipeline import run_fase1
            run_fase1()
            if self.cancel_event.is_set():
                self.root.after(0, lambda: self._on_fase1_error("Cancelado"))
                return
            self.root.after(0, self._on_fase1_completed)
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda msg=msg: self._on_fase1_error(msg))
        finally:
            self.root.after(0, self._release_fase1_ui)

    def _on_fase1_completed(self):
        self._log_message("✓ Fase 1 completada. Exportando Excel en carpeta de estudio de mercado...")
        self._refresh_checkpoint_label()
        from etl.config import ESTUDIO_MERCADO_DIR
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ruta = ESTUDIO_MERCADO_DIR / f"Base_Programas_Categoria_F1_{ts}.xlsx"
        self._log_message(f"Exportando Excel → {ruta} ...")

        def _export_worker():
            try:
                from etl.mercado_pipeline import exportar_base_maestra_excel

                resultado = exportar_base_maestra_excel(ruta_salida=ruta)

                def _ok():
                    self._log_message(f"✓ Excel generado: {resultado.name}")
                    self._refresh_checkpoint_label()
                    if messagebox.askyesno("Fase 1 completada", f"Excel guardado en:\n{resultado}\n\n¿Deseas abrirlo ahora?", parent=self.root):
                        try:
                            import os

                            os.startfile(str(resultado))
                        except Exception as exc:
                            safe_messagebox_error("Error", f"No se pudo abrir:\n{exc}", parent=self.root)

                self.root.after(0, _ok)
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda msg=msg: self._log_message(f"✗ Error al exportar: {msg}"))
            finally:
                self.root.after(0, lambda: self.btn_fase1.config(state=tk.NORMAL))

        threading.Thread(target=_export_worker, daemon=True).start()

    def _on_fase1_error(self, error_msg: str):
        if "Cancelado" in error_msg or self.cancel_event.is_set():
            self.progress_label_fase1.config(text="Cancelado", foreground=EAFIT["warning"])
            self._log_message("✗ Fase 1 cancelada.")
            messagebox.showinfo("Cancelado", "La Fase 1 fue cancelada.", parent=self.root)
        else:
            self.progress_label_fase1.config(text="Error", foreground=EAFIT["danger"])
            self._log_message(f"✗ ERROR Fase 1: {error_msg}")
            safe_messagebox_error("Error Fase 1", f"Error durante la Fase 1:\n\n{error_msg}", parent=self.root)
        self._refresh_checkpoint_label()

    def _release_fase1_ui(self):
        try:
            self.progress_fase1.stop()
        except Exception:
            pass
        self.is_running = False
        self.btn_cancel_fase1.config(state=tk.DISABLED)
        self._mercado_set_secondary_busy(False)
        self.btn_fase1.config(state=tk.NORMAL)
        self.btn_execute.config(state=tk.NORMAL)
        self.progress_label_fase1.config(text="")


class EstudioMercadoResultsPage(ttk.Frame):
    """Página dedicada para ver y editar el resultado del estudio de mercado (Estudio_Mercado_Colombia.xlsx)."""

    def __init__(self, parent: tk.Misc, on_back=None):
        super().__init__(parent)
        self.on_back = on_back
        self.root = parent.winfo_toplevel()
        from etl.config import ARCHIVO_ESTUDIO_MERCADO
        self.file_path = ARCHIVO_ESTUDIO_MERCADO
        self.active_sheet = "total"
        self.df_total = None
        self.df_detalle = None
        self.df_eafit = None
        self._filtered_df = None
        self.page_size = 200
        self.page_index = 0
        self.pending_updates = {}
        self.editable_columns = {
            "CATEGORIA_FINAL",
            "FUENTE_CATEGORIA",
            "calificacion_final",
            "REQUIERE_REVISION",
        }
        # Inicializar lista de categorías (se rellena cuando se carga el Excel)
        self._lista_categorias: list[str] = []
        self._dropdown_vals: dict[str, list[str]] = {
            "FUENTE_CATEGORIA": ["CRUCE_SNIES", "MATCH_NOMBRE", "MATCH_CATEGORIA", "KNN_TFIDF", "MANUAL", "PIPELINE"],
            "REQUIERE_REVISION": ["True", "False"],
            "CATEGORIA_FINAL": [],
        }
        self._setup_ui()
        self._load()

    def _on_resize(self, w: int, h: int) -> None:
        """Ajusta tabla y área de mensajes según espacio disponible."""
        try:
            table_pixels = max(150, h - 340)
            if self.table:
                self.table.set_height_from_pixels(table_pixels)
        except (tk.TclError, AttributeError):
            pass
        try:
            log_lines = max(3, min(6, (h - 500) // 20))
            self.msg.config(height=log_lines)
        except (tk.TclError, AttributeError):
            pass

    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding=20, style="Page.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main_frame, style="Page.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        header_left = ttk.Frame(header, style="Page.TFrame")
        header_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(header_left, text="📊 Resultados — Estudio de Mercado Colombia", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header_left, text="Vista y edición controlada del resultado generado.", style="Muted.TLabel").pack(anchor="w", pady=(4, 0), fill=tk.X)
        if self.on_back:
            ttk.Button(header, text="← Volver", command=self._on_back_clicked, style="Back.TButton").pack(side=tk.RIGHT)

        # Fila 1: botones
        row1 = ttk.Frame(main_frame, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row1, text="🔄 Recargar", command=self._load).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row1, text="📂 Abrir en Excel", command=self._open_excel).pack(side=tk.LEFT, padx=6)
        self.btn_integridad = ttk.Button(
            row1,
            text="🔍 Verificar integridad",
            command=self._verificar_integridad,
            style="Secondary.TButton",
        )
        self.btn_integridad.pack(side=tk.LEFT, padx=6)
        self.btn_save = ttk.Button(row1, text="💾 Guardar cambios", command=self._save, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="↩️ Descartar todo", command=self._discard_all).pack(side=tk.LEFT, padx=6)

        # Fila 2: pestañas de hoja (por defecto programas_detalle activa → Primary)
        row2 = ttk.Frame(main_frame, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=(0, 8))
        self.btn_sheet_total = ttk.Button(row2, text="📋 Resumen (total)", command=lambda: self._switch_sheet("total"), style="Secondary.TButton")
        self.btn_sheet_total.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_sheet_datos = ttk.Button(
            row2,
            text="📊 Datos completos",
            command=lambda: self._switch_sheet("total_tabla"),
            style="Secondary.TButton",
        )
        self.btn_sheet_datos.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_sheet_detalle = ttk.Button(row2, text="📄 Programas detalle", command=lambda: self._switch_sheet("programas_detalle"), style="Primary.TButton")
        self.btn_sheet_detalle.pack(side=tk.LEFT, padx=6)
        self.btn_sheet_eafit = ttk.Button(
            row2,
            text="🎓 EAFIT vs Mercado",
            command=lambda: self._switch_sheet("eafit"),
            style="Secondary.TButton",
        )
        self.btn_sheet_eafit.pack(side=tk.LEFT, padx=(6, 0))

        # Fila 3: filtros
        row3 = ttk.Frame(main_frame, style="Card.TFrame")
        row3.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row3, text="Buscar:").pack(side=tk.LEFT, padx=(0, 6))
        self.search_var = tk.StringVar(value="")
        ttk.Entry(row3, textvariable=self.search_var, width=30).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="Buscar", command=self._apply_filter).pack(side=tk.LEFT, padx=6)
        self.calif_label = ttk.Label(row3, text="Calificación:")
        self.calif_label.pack(side=tk.LEFT, padx=(14, 6))
        self.filter_calif_var = tk.StringVar(value="TODAS")
        self.filter_calif = ttk.Combobox(
            row3, textvariable=self.filter_calif_var, state="readonly",
            values=["TODAS", "Verde (≥4)", "Amarillo (≥3)", "Rojo (<3)"], width=14,
        )
        self.filter_calif.pack(side=tk.LEFT, padx=2)
        self.filter_calif.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())
        ttk.Button(row3, text="Limpiar filtros", command=self._clear_filters).pack(side=tk.LEFT, padx=6)

        # Paginador
        pager = ttk.Frame(main_frame, style="Card.TFrame")
        pager.pack(fill=tk.X, pady=(0, 6))
        self.page_label = ttk.Label(pager, text="Página: -", foreground=EAFIT["text_muted"])
        self.page_label.pack(side=tk.LEFT)
        self.pending_label = ttk.Label(pager, text="Cambios pendientes: 0", foreground=EAFIT["text_muted"])
        self.pending_label.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(pager, text="Anterior", command=self._prev_page).pack(side=tk.RIGHT)
        ttk.Button(pager, text="Siguiente", command=self._next_page).pack(side=tk.RIGHT, padx=6)

        # Banner solo lectura (oculto por defecto)
        self.readonly_banner = ttk.Label(self, text="", foreground=EAFIT["danger"], font=("Segoe UI", 9, "bold"))
        self.readonly_banner.pack(fill=tk.X, padx=10, pady=(0, 6))

        # Área de mensajes
        self.msg = tk.Text(self, height=4, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9), bg=EAFIT["card_bg"], fg=EAFIT["text"])
        self.msg.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Placeholder para tabla (se reconstruye en _switch_sheet)
        self.table_frame = ttk.Frame(main_frame, style="Card.TFrame")
        self.table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.table = None

    def _on_back_clicked(self):
        if self.pending_updates:
            if not messagebox.askyesno("Cambios pendientes", "¿Descartar cambios sin guardar?", parent=self.root):
                return
        if self.on_back:
            self.on_back()

    def _log(self, message: str):
        self.msg.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.msg.insert(tk.END, f"[{ts}] {message}\n")
        self.msg.see(tk.END)
        self.msg.config(state=tk.DISABLED)

    def _load(self):
        if not self.file_path.exists():
            self._log("⚠️ No existe el archivo. Ejecuta el pipeline primero.")
            return
        try:
            from etl.exceptions_helpers import leer_excel_con_reintentos
            df_try = leer_excel_con_reintentos(self.file_path, sheet_name="total", header=0)
            if len(df_try.columns) > 0 and str(df_try.columns[0]).strip() == "CATEGORIA_FINAL":
                self.df_total = df_try
                # El Excel de 'total' tiene un sub-encabezado con CATEGORIA_FINAL = NaN.
                # Filtramos esa fila para que no aparezca "nan" en la GUI.
                if "CATEGORIA_FINAL" in self.df_total.columns:
                    self.df_total = (
                        self.df_total[self.df_total["CATEGORIA_FINAL"].notna()]
                        .reset_index(drop=True)
                    )
            else:
                self.df_total = leer_excel_con_reintentos(self.file_path, sheet_name="total", header=1)
                # Con header=1, la primera columna suele venir como "Unnamed: 0".
                if "Unnamed: 0" in self.df_total.columns and "CATEGORIA_FINAL" not in self.df_total.columns:
                    self.df_total = self.df_total.rename(columns={"Unnamed: 0": "CATEGORIA_FINAL"})
            self.df_detalle = leer_excel_con_reintentos(self.file_path, sheet_name="programas_detalle")
            # Hoja opcional (Fase 6): si no existe, no bloquea la UI.
            try:
                self.df_eafit = leer_excel_con_reintentos(self.file_path, sheet_name="eafit_vs_mercado")
            except Exception:
                self.df_eafit = None
        except Exception as e:
            self._log(f"✗ Error al leer: {e}")
            return
        self.pending_updates.clear()
        # Poblar lista de categorías desde hoja total (para dropdown CATEGORIA_FINAL)
        try:
            if self.df_total is not None and getattr(self.df_total, "columns", None) is not None and len(self.df_total.columns) > 0:
                cat_col_name = str(self.df_total.columns[0])
                cats = (
                    self.df_total[cat_col_name]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                cats = sorted(cats)
                self._lista_categorias = cats
                self._dropdown_vals["CATEGORIA_FINAL"] = cats
                # Propagar al dropdown_values de la tabla actual (si ya existe)
                if self.table and hasattr(self.table, "dropdown_values"):
                    self.table.dropdown_values["CATEGORIA_FINAL"] = cats
        except Exception as e:
            self._log(f"⚠️ No se pudo poblar dropdown de categorías: {e}")
        self._switch_sheet(self.active_sheet)
        n_total = len(self.df_total) if self.df_total is not None else 0
        n_detalle = len(self.df_detalle) if self.df_detalle is not None else 0
        self._log(f"✓ Archivo cargado: {n_total} categorías, {n_detalle} programas")

    def _switch_sheet(self, sheet_name: str):
        self.active_sheet = sheet_name
        self.page_index = 0
        # Reset estado de tabla (si existía)
        self.table = None
        self._filtered_df = None

        # Estilos botones
        if hasattr(self, "btn_sheet_total"):
            self.btn_sheet_total.config(style="Secondary.TButton")
        if hasattr(self, "btn_sheet_datos"):
            self.btn_sheet_datos.config(style="Secondary.TButton")
        if hasattr(self, "btn_sheet_detalle"):
            self.btn_sheet_detalle.config(style="Secondary.TButton")
        if hasattr(self, "btn_sheet_eafit"):
            self.btn_sheet_eafit.config(style="Secondary.TButton")

        # ── Hoja resumida (panel visual, no editable) ─────────────────────
        if sheet_name == "total":
            self.btn_sheet_total.config(style="Primary.TButton")
            self.readonly_banner.config(text="")
            # Ocultar filtros
            self.calif_label.pack_forget()
            self.filter_calif.pack_forget()
            self._build_resumen_panel()
            return

        # ── Hoja tabla (total editable) ───────────────────────────────────
        if sheet_name == "total_tabla":
            self.btn_sheet_datos.config(style="Primary.TButton")
            self.readonly_banner.config(text="")
            self.calif_label.pack(side=tk.LEFT, padx=(14, 6))
            self.filter_calif.pack(side=tk.LEFT, padx=2)

            display_cols = [
                "CATEGORIA_FINAL", "FUENTE_CATEGORIA", "calificacion_final",
                "suma_matricula_2024", "AAGR_suma", "participacion_2024",
                "salario_promedio", "pct_no_matriculados_2024",
                "num_programas_2024", "costo_promedio",
            ]
            df = self.df_total
            editable = self.editable_columns

        # ── Hoja programas detalle (editable) ──────────────────────────────
        elif sheet_name == "programas_detalle":
            self.btn_sheet_detalle.config(style="Primary.TButton")
            self.readonly_banner.config(text="")
            self.calif_label.pack_forget()
            self.filter_calif.pack_forget()

            display_cols = [
                "CÓDIGO_SNIES_DEL_PROGRAMA", "NOMBRE_DEL_PROGRAMA", "NOMBRE_INSTITUCIÓN", "NIVEL_DE_FORMACIÓN",
                "CATEGORIA_FINAL", "FUENTE_CATEGORIA", "REQUIERE_REVISION", "calificacion_final", "PROBABILIDAD",
                "ESTADO_PROGRAMA", "ACTIVO_PIPELINE",
            ]
            df = self.df_detalle
            editable = self.editable_columns

        # ── Hoja EAFIT vs Mercado (opcional, solo lectura) ───────────────
        elif sheet_name == "eafit":
            self.btn_sheet_eafit.config(style="Primary.TButton")
            self.readonly_banner.config(text="")
            self.calif_label.pack_forget()
            self.filter_calif.pack_forget()

            if self.df_eafit is None:
                for w in self.table_frame.winfo_children():
                    w.destroy()
                tk.Label(
                    self.table_frame,
                    text=(
                        "⚠️ La hoja 'eafit_vs_mercado' no existe aún.\n\n"
                        "Coloca 'programas_para_valorizacion.xlsx' en:\n"
                        "ref/backup/\n\n"
                        "y ejecuta el pipeline para generarla."
                    ),
                    font=("Segoe UI", 10),
                    fg=EAFIT["text_muted"],
                    bg=EAFIT["bg"],
                    justify="center",
                ).pack(expand=True)
                return

            display_cols = [
                "PROGRAMA_EAFIT",
                "NIVEL_FORMACION",
                "TIENE_ESTUDIO_MERCADO",
                "CATEGORIA_MERCADO",
                "SEMAFORO_CALIDAD",
                "OPORTUNIDAD",
                "calificacion_final",
                "AAGR_PCT",
                "suma_matricula_2024",
                "salario_promedio",
                "num_programas_2024",
                "costo_promedio",
            ]
            df = self.df_eafit
            editable = set()

        else:
            # Fallback: mostrar detalle
            self.btn_sheet_detalle.config(style="Primary.TButton")
            display_cols = [
                "CÓDIGO_SNIES_DEL_PROGRAMA", "NOMBRE_DEL_PROGRAMA", "NOMBRE_INSTITUCIÓN", "NIVEL_DE_FORMACIÓN",
                "CATEGORIA_FINAL", "FUENTE_CATEGORIA", "REQUIERE_REVISION", "calificacion_final", "PROBABILIDAD",
                "ESTADO_PROGRAMA", "ACTIVO_PIPELINE",
            ]
            df = self.df_detalle
            editable = self.editable_columns

        # ── Construcción de tabla (total_tabla, programas_detalle, eafit) ──
        for w in self.table_frame.winfo_children():
            w.destroy()
        if df is not None:
            display_cols = [c for c in display_cols if c in df.columns]
        if not display_cols:
            display_cols = list(df.columns)[:10] if df is not None and len(df.columns) else []

        self.table = EditableTable(
            self.table_frame,
            columns=display_cols,
            height=18,
            editable_columns=editable,
            on_change=self._on_cell_change,
            dropdown_values=self._dropdown_vals,
        )
        self.table.pack(fill=tk.BOTH, expand=True)
        self._apply_filter()

    def _build_resumen_panel(self) -> None:
        """Crea un panel visual con KPIs/rankings para la vista 'total'."""
        import numpy as np

        for w in self.table_frame.winfo_children():
            w.destroy()

        det = self.df_detalle
        tot = self.df_total
        if det is None or tot is None:
            self._resumen_panel = None
            ttk.Label(self.table_frame, text="Primero carga el archivo.", foreground=EAFIT["text_muted"]).pack(expand=True)
            return

        # Canvas + scrollbar para contenido desplazable
        outer = ttk.Frame(self.table_frame)
        outer.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, bg=EAFIT["card_bg"], highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Mantener ancho del contenido dentro del canvas
            try:
                canvas.itemconfig(canvas_window, width=canvas.winfo_width())
            except Exception:
                pass

        inner.bind("<Configure>", _on_configure)

        def _on_mousewheel(e):
            # Windows: e.delta suele ser múltiplo de 120
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

        def _safe_sum(df: pd.DataFrame, col: str) -> float:
            if col not in df.columns:
                return 0.0
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

        mat19 = _safe_sum(det, "matricula_2019")
        mat24 = _safe_sum(det, "matricula_2024")
        crecimiento = ((mat24 - mat19) / mat19) if mat19 else 0.0

        total_programas = int(len(det))
        total_categorias = int(len(tot))

        # Activos / con matrícula 2024 (si existen columnas)
        if "es_activo" in det.columns:
            es_activo_sum = int(det["es_activo"].fillna(0).astype(bool).sum())
        else:
            es_activo_sum = 0
        if "tiene_matricula_2024" in det.columns:
            tiene_matricula_2024 = int(det["tiene_matricula_2024"].fillna(False).astype(bool).sum())
        else:
            tiene_matricula_2024 = int(pd.to_numeric(det.get("matricula_2024", 0), errors="coerce").fillna(0).gt(0).sum())

        # tot["calificacion_final"] debe ser una Series; si la columna no existe,
        # usamos una Series vacía para evitar errores tipo `len(float)`.
        if "calificacion_final" in tot.columns:
            calif = pd.to_numeric(tot["calificacion_final"], errors="coerce")
        else:
            calif = pd.Series([], dtype=float)
        verdes = int((calif >= 4.0).sum()) if len(calif) else 0
        amarillos = int(((calif >= 3.0) & (calif < 4.0)).sum()) if len(calif) else 0
        rojos = int((calif < 3.0).sum()) if len(calif) else 0
        calif_prom = float(calif.mean()) if len(calif) else 0.0

        req_revision = int(det.get("REQUIERE_REVISION", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum())

        # Certeza por fuente (CRUCE_SNIES + MATCH_NOMBRE)
        fuentes = det.get("FUENTE_CATEGORIA", pd.Series([], dtype=object)).astype(str).str.upper().str.strip()
        cruce_snies = int(fuentes.eq("CRUCE_SNIES").sum())
        match_nombre = int(fuentes.eq("MATCH_NOMBRE").sum())
        certeza_100 = ((cruce_snies + match_nombre) / total_programas * 100.0) if total_programas else 0.0

        # Top 5 rankings desde tot/ag
        def _top_rows(col: str, n: int, ascending: bool = False):
            if col not in tot.columns:
                return []
            df2 = tot.sort_values(col, ascending=ascending).head(n)
            cat_col = tot.columns[0] if len(tot.columns) else "CATEGORIA_FINAL"
            return [(str(r[cat_col]), r.get(col), r.get("calificacion_final", "")) for _, r in df2.iterrows()]

        cat_col_name = tot.columns[0] if len(tot.columns) else "CATEGORIA_FINAL"

        top_mat = []
        if "suma_matricula_2024" in tot.columns:
            tmp = tot.sort_values("suma_matricula_2024", ascending=False).head(5)
            top_mat = [(str(r[cat_col_name]), r.get("suma_matricula_2024"), r.get("calificacion_final", "")) for _, r in tmp.iterrows()]

        top_aagr = []
        if "AAGR_suma" in tot.columns:
            tmp = tot.sort_values("AAGR_suma", ascending=False).head(5)
            top_aagr = [(str(r[cat_col_name]), r.get("AAGR_suma"), r.get("suma_matricula_2024", "")) for _, r in tmp.iterrows()]

        top_salario = []
        if "salario_promedio" in tot.columns:
            tmp = tot.sort_values("salario_promedio", ascending=False).head(5)
            top_salario = [(str(r[cat_col_name]), r.get("salario_promedio"), r.get("salario_proyectado_pesos_hoy", "")) for _, r in tmp.iterrows()]

        worst_aagr = []
        if "AAGR_suma" in tot.columns:
            tmp = tot.sort_values("AAGR_suma", ascending=True).head(5)
            worst_aagr = [(str(r[cat_col_name]), r.get("AAGR_suma"), r.get("suma_matricula_2024", "")) for _, r in tmp.iterrows()]

        # Calidad por fuente
        fuentes_counts = fuentes.value_counts()
        total_fuentes = int(len(fuentes)) if len(fuentes) else 0
        def _pct(count: int) -> str:
            return f"{(count/total_fuentes*100.0):.1f}%" if total_fuentes else "0.0%"

        quality_rows = [
            ("CRUCE_SNIES", int(fuentes_counts.get("CRUCE_SNIES", 0)), _pct(int(fuentes_counts.get("CRUCE_SNIES", 0)))),
            ("MATCH_NOMBRE", int(fuentes_counts.get("MATCH_NOMBRE", 0)), _pct(int(fuentes_counts.get("MATCH_NOMBRE", 0)))),
            ("KNN_TFIDF", int(fuentes_counts.get("KNN_TFIDF", 0)), _pct(int(fuentes_counts.get("KNN_TFIDF", 0)))),
            ("Requiere revisión", req_revision, _pct(req_revision)),
        ]

        # Secciones
        header = ttk.Frame(inner, padding=(6, 10))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="ESTUDIO DE MERCADO — COLOMBIA",
            font=("Segoe UI", 16, "bold"),
            foreground=EAFIT["azul_zafre"],
        ).pack(anchor="w")

        subtitle = ttk.Label(header, text="KPIs globales y rankings (vista resumida)", foreground=EAFIT["text_muted"])
        subtitle.pack(anchor="w", pady=(4, 0))

        # Bloque KPIs (grid)
        kpi_frame = ttk.Frame(inner, padding=(6, 8))
        kpi_frame.pack(fill=tk.X)
        kpis = [
            ("Total programas analizados", total_programas),
            ("Total categorías", total_categorias),
            ("Matrícula total 2024", mat24),
            ("Matrícula total 2019", mat19),
            ("Crecimiento global 2019→2024", f"{crecimiento*100:.1f}%"),
            ("Programas activos", es_activo_sum),
            ("Programas con matrícula 2024", tiene_matricula_2024),
            ("Calificación promedio", f"{calif_prom:.2f}"),
        ]
        for i, (lab, val) in enumerate(kpis):
            r = i // 2
            c = i % 2
            card = ttk.Frame(kpi_frame, padding=10, style="Card.TFrame")
            card.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            ttk.Label(card, text=lab, foreground=EAFIT["text_muted"]).pack(anchor="w")
            ttk.Label(card, text=str(val), font=("Consolas", 12, "bold")).pack(anchor="w", pady=(6, 0))

        # Semáforo de categorías por calidad
        quality_frame = ttk.Frame(inner, padding=(6, 8))
        quality_frame.pack(fill=tk.X)
        ttk.Label(quality_frame, text="Semáforo de calidad (por calificación en total)", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        sem = ttk.Frame(quality_frame)
        sem.pack(fill=tk.X)

        def _color_box(parent, title: str, count: int, color: str):
            b = tk.Label(parent, text=f"{title}: {count}", bg=color, fg="#000000", font=("Consolas", 10, "bold"), padx=10, pady=8)
            b.pack(side=tk.LEFT, padx=(0, 8))

        _color_box(sem, "VERDE (>=4.0)", verdes, "#C6EFCE")
        _color_box(sem, "AMARILLO (3.0-3.9)", amarillos, "#FFEB9C")
        _color_box(sem, "ROJO (<3.0)", rojos, "#FFC7CE")

        # Certeza y revisión
        cer_frame = ttk.Frame(inner, padding=(6, 8))
        cer_frame.pack(fill=tk.X)
        ttk.Label(cer_frame, text="Certeza de clasificación y revisión manual", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        cmsg = f"Certeza 100% (CRUCE_SNIES + MATCH_NOMBRE): {certeza_100:.1f}%"
        ttk.Label(cer_frame, text=cmsg, font=("Consolas", 10, "bold")).pack(anchor="w")
        ttk.Label(cer_frame, text=f"Requieren revisión manual: {req_revision:,}", foreground=EAFIT["text_muted"]).pack(anchor="w", pady=(6, 0))

        # Rankings
        def _section(title: str):
            sec = ttk.Frame(inner, padding=(6, 10))
            sec.pack(fill=tk.X, pady=(8, 0))
            ttk.Label(sec, text=title, font=("Segoe UI", 12, "bold")).pack(anchor="w")
            return sec

        def _list_rows(parent, rows: list[tuple], headers: list[str]):
            # Headers
            head = ttk.Frame(parent)
            head.pack(fill=tk.X, pady=(6, 0))
            for i, h in enumerate(headers):
                tk.Label(head, text=h, bg="#000066", fg="white", font=("Segoe UI", 9, "bold"), padx=6, pady=4).grid(
                    row=0, column=i, sticky="nsew", padx=(0, 2)
                )
            # Rows
            for r in rows:
                rowf = ttk.Frame(parent, padding=(0, 4))
                rowf.pack(fill=tk.X)
                for i, v in enumerate(r):
                    ttk.Label(rowf, text=str(v) if v is not None else "", font=("Consolas", 9)).grid(
                        row=0, column=i, sticky="w", padx=(0, 10)
                    )

        sec1 = _section("Top 5 — Mayor matrícula 2024")
        _list_rows(sec1, [(a, f"{b:,.0f}", f"{c}") for a, b, c in top_mat] if top_mat else [("-", "-", "-")], ["Categoría", "Matrícula 2024", "Calificación"])

        sec2 = _section("Top 5 — Mayor crecimiento (AAGR)")
        _list_rows(sec2, [(a, f"{b:.3f}", c) for a, b, c in top_aagr] if top_aagr else [("-", "-", "-")], ["Categoría", "AAGR", "Matrícula 2024"])

        sec3 = _section("Top 5 — Mejor salario (SMLMV)")
        if top_salario:
            _list_rows(
                sec3,
                [(a, b, c) for a, b, c in top_salario],
                ["Categoría", "Salario SMLMV", "Salario pesos hoy"],
            )
        else:
            ttk.Label(sec3, text="Sin datos de salario.", foreground=EAFIT["text_muted"]).pack(anchor="w", pady=(6, 0))

        sec4 = _section("Top 5 — Menor crecimiento (AAGR)")
        _list_rows(sec4, [(a, f"{b:.3f}", c) for a, b, c in worst_aagr] if worst_aagr else [("-", "-", "-")], ["Categoría", "AAGR", "Matrícula 2024"])

        # Calidad por fuente
        sec5 = _section("Calidad de clasificación (por fuente)")
        src_rows = []
        for name, count, pct in quality_rows:
            conf = "100% — cruce exacto" if name == "CRUCE_SNIES" else "100% — match exacto" if name == "MATCH_NOMBRE" else "Variable" if name == "KNN_TFIDF" else "—"
            src_rows.append((name, count, pct, conf))
        _list_rows(sec5, src_rows if src_rows else [("-", "-", "-", "-")], ["Fuente", "Programas", "% del total", "Confianza"])

    def _apply_filter(self):
        # La vista 'total' es un panel (no usa tabla/paginación)
        if self.active_sheet == "total":
            self._filtered_df = None
            return

        if self.active_sheet == "total_tabla":
            df = self.df_total
        elif self.active_sheet == "eafit":
            df = self.df_eafit
        else:
            df = self.df_detalle
        if df is None:
            self._filtered_df = None
            self._render_page()
            return
        df = df.copy()
        search = (self.search_var.get() or "").strip()
        if search:
            search_lower = search.lower()
            mask = pd.Series(False, index=df.index)
            for col in df.select_dtypes(include=["object", "string"]).columns:
                mask |= df[col].astype(str).str.lower().str.contains(search_lower, na=False)
            df = df.loc[mask]
        if self.active_sheet == "total_tabla" and "calificacion_final" in df.columns:
            calif = self.filter_calif_var.get()
            if calif == "Verde (≥4)":
                df = df[df["calificacion_final"].astype(float) >= 4.0]
            elif calif == "Amarillo (≥3)":
                df = df[(df["calificacion_final"].astype(float) >= 3.0) & (df["calificacion_final"].astype(float) < 4.0)]
            elif calif == "Rojo (<3)":
                df = df[df["calificacion_final"].astype(float) < 3.0]
        self._filtered_df = df
        self.page_index = 0
        self._render_page()

    def _clear_filters(self):
        self.search_var.set("")
        self.filter_calif_var.set("TODAS")
        self._apply_filter()

    def _render_page(self):
        if self._filtered_df is None:
            if self.table:
                self.table.set_rows([])
            self.page_label.config(text="Página: -")
            self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
            return
        total = len(self._filtered_df)
        if total == 0:
            if self.table:
                self.table.set_rows([])
            self.page_label.config(text="Página: 0/0  (0 filas totales)")
            self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
            return
        max_pages = max(1, (total + self.page_size - 1) // self.page_size)
        self.page_index = max(0, min(self.page_index, max_pages - 1))
        start = self.page_index * self.page_size
        end = min(total, start + self.page_size)
        df_page = self._filtered_df.iloc[start:end].copy()
        if self.pending_updates:
            for i in range(len(df_page)):
                if self.active_sheet in ("total", "total_tabla"):
                    cat = df_page.iloc[i].get("CATEGORIA_FINAL")
                    key = str(cat).strip() if cat is not None else None
                else:
                    snies = df_page.iloc[i].get("CÓDIGO_SNIES_DEL_PROGRAMA")
                    key = str(snies).strip() if snies is not None else None
                if key and key in self.pending_updates:
                    for k, v in self.pending_updates[key].items():
                        if k in df_page.columns:
                            df_page.at[df_page.index[i], k] = v
        rows = df_page.to_dict(orient="records")
        if self.table:
            self.table.set_rows(rows)
        self.page_label.config(text=f"Página: {self.page_index + 1} de {max_pages}  ({total} filas totales)")
        self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
        if self.pending_updates and self.btn_save:
            self.btn_save.config(state=tk.NORMAL)

    def _on_cell_change(self, idx: int, column: str, new_value: str):
        rows = self.table.get_rows() if self.table else []
        if idx < 0 or idx >= len(rows):
            return
        row = rows[idx]

        # Validación: calificacion_final debe ser float entre 1.0 y 5.0
        if column == "calificacion_final":
            try:
                v = float(new_value)
            except (ValueError, TypeError):
                self._revert_cell(idx, column)
                self._log("⚠️ calificacion_final debe ser un número entre 1.0 y 5.0")
                return
            if v < 1.0 or v > 5.0:
                self._revert_cell(idx, column)
                self._log("⚠️ calificacion_final debe estar entre 1.0 y 5.0")
                return

        # Clave de identificación según hoja activa
        if self.active_sheet in ("total", "total_tabla"):
            categoria_final = row.get("CATEGORIA_FINAL")
            if categoria_final is None:
                return
            key = str(categoria_final).strip()
        else:
            snies = row.get("CÓDIGO_SNIES_DEL_PROGRAMA")
            if snies is None:
                return
            key = str(snies).strip()

        if not key:
            return

        if key not in self.pending_updates:
            self.pending_updates[key] = {}

        self.pending_updates[key][column] = new_value

        # Si cambia la categoría, auto-marcar como MANUAL y limpiar REQUIERE_REVISION
        if column == "CATEGORIA_FINAL":
            self.pending_updates[key]["FUENTE_CATEGORIA"] = "MANUAL"
            # Independencia de edits: si el usuario ya cambió REQUIERE_REVISION en la sesión,
            # no lo sobreescribimos silenciosamente.
            if "REQUIERE_REVISION" not in self.pending_updates[key]:
                self.pending_updates[key]["REQUIERE_REVISION"] = False
            self._log(
                "✓ Categoría cambiada → FUENTE_CATEGORIA=MANUAL "
                "(REQUIERE_REVISION se conserva si fue editado)"
            )

        # Si cambia FUENTE_CATEGORIA a MANUAL, también limpiar REQUIERE_REVISION
        if column == "FUENTE_CATEGORIA" and str(new_value).strip().upper() == "MANUAL":
            if "REQUIERE_REVISION" not in self.pending_updates[key]:
                self.pending_updates[key]["REQUIERE_REVISION"] = False

        self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
        self.btn_save.config(state=tk.NORMAL)

    def _revert_cell(self, idx: int, column: str):
        """Restaura el valor de una celda desde _filtered_df (valor mostrado antes del cambio rechazado)."""
        if self.table is None or self._filtered_df is None:
            return
        start = self.page_index * self.page_size
        if start + idx >= len(self._filtered_df):
            return
        orig_val = self._filtered_df.iloc[start + idx].get(column)
        self.table.set_cell_value(idx, column, str(orig_val) if orig_val is not None and not (isinstance(orig_val, float) and pd.isna(orig_val)) else "")

    def _verificar_integridad(self):
        """
        Verifica la integridad de los datos y muestra un informe.
        Comprueba: categorías nulas, FUENTE_CATEGORIA inválida, calificacion fuera de rango,
        SNIES duplicados, ACTIVO_PIPELINE inconsistente, categorías huérfanas,
        y consistencia semántica de MANUAL con REQUIERE_REVISION.
        """
        if self.df_detalle is None or self.df_total is None:
            messagebox.showwarning("Sin datos", "Primero carga el archivo.", parent=self.root)
            return

        det = self.df_detalle.copy()
        tot = self.df_total.copy()
        problemas: list[str] = []
        ok_msgs: list[str] = []

        # 1. Programas sin CATEGORIA_FINAL
        n = det["CATEGORIA_FINAL"].isna().sum() if "CATEGORIA_FINAL" in det.columns else 0
        if n > 0:
            problemas.append(f"⚠️ {n:,} programas sin CATEGORIA_FINAL (nulo/vacío)")
        else:
            ok_msgs.append("✓ Todos los programas tienen CATEGORIA_FINAL")

        # 2. FUENTE_CATEGORIA con valores inválidos
        valores_validos_fuente = {"CRUCE_SNIES", "MATCH_NOMBRE", "MATCH_CATEGORIA", "KNN_TFIDF", "MANUAL", "PIPELINE"}
        if "FUENTE_CATEGORIA" in det.columns:
            invalidos = det[
                ~det["FUENTE_CATEGORIA"].astype(str).isin(valores_validos_fuente | {"nan", ""})
            ]
            invalidos = invalidos.dropna(subset=["FUENTE_CATEGORIA"])
            if len(invalidos) > 0:
                vals = invalidos["FUENTE_CATEGORIA"].value_counts().to_dict()
                problemas.append(f"⚠️ {len(invalidos):,} programas con FUENTE_CATEGORIA inválida: {vals}")
            else:
                ok_msgs.append("✓ FUENTE_CATEGORIA con valores válidos")

        # 3. calificacion_final fuera de [1, 5] en hoja total
        if "calificacion_final" in tot.columns:
            cal = pd.to_numeric(tot["calificacion_final"], errors="coerce")
            n_out = int(((cal < 1) | (cal > 5)).sum())
            n_nan = int(cal.isna().sum())
            if n_out > 0:
                problemas.append(f"⚠️ {n_out:,} categorías con calificacion_final fuera de [1, 5]")
            elif n_nan > 0:
                problemas.append(f"⚠️ {n_nan:,} categorías con calificacion_final nula")
            else:
                ok_msgs.append(f"✓ calificacion_final en rango [1, 5] para las {len(tot):,} categorías")

        # 4. SNIES duplicados en programas_detalle
        if "CÓDIGO_SNIES_DEL_PROGRAMA" in det.columns:
            dups = int(det["CÓDIGO_SNIES_DEL_PROGRAMA"].astype(str).duplicated().sum())
            if dups > 0:
                problemas.append(f"⚠️ {dups:,} códigos SNIES duplicados en programas_detalle")
            else:
                ok_msgs.append("✓ Sin códigos SNIES duplicados")

        # 5. ACTIVO_PIPELINE inconsistente
        if (
            "ACTIVO_PIPELINE" in det.columns
            and "matricula_2024" in det.columns
            and "ESTADO_PROGRAMA" in det.columns
        ):
            mat = pd.to_numeric(det["matricula_2024"], errors="coerce").fillna(0)
            estado_act = det["ESTADO_PROGRAMA"].astype(str).str.lower().str.strip() == "activo"
            activo_flag = det["ACTIVO_PIPELINE"].astype(str).str.lower().isin(["true", "1", "yes"])
            inconsistentes = det[activo_flag & (mat == 0) & (~estado_act)]
            if len(inconsistentes) > 0:
                problemas.append(
                    f"⚠️ {len(inconsistentes):,} programas marcados ACTIVO_PIPELINE=True "
                    "pero sin matrícula 2024 ni estado 'activo'"
                )
            else:
                ok_msgs.append("✓ ACTIVO_PIPELINE coherente con matrícula y estado")

        # 6. Categorías en total sin ningún programa en detalle
        if "CATEGORIA_FINAL" in det.columns and len(tot.columns) > 0:
            cat_col_name = str(tot.columns[0])
            cats_detalle = set(det["CATEGORIA_FINAL"].dropna().astype(str).unique())
            cats_total = set(tot[cat_col_name].dropna().astype(str).unique())
            huerfanas = cats_total - cats_detalle
            if huerfanas:
                ejemplos = ", ".join(sorted(huerfanas)[:5])
                problemas.append(
                    f"⚠️ {len(huerfanas):,} categorías en hoja 'total' sin programas en detalle: "
                    f"{ejemplos}" + ("..." if len(huerfanas) > 5 else "")
                )
            else:
                ok_msgs.append("✓ Todas las categorías de 'total' tienen programas en detalle")

        # 7. Programas MANUAL con REQUIERE_REVISION=True
        if "FUENTE_CATEGORIA" in det.columns and "REQUIERE_REVISION" in det.columns:
            manuales = det[det["FUENTE_CATEGORIA"].astype(str).str.upper().str.strip() == "MANUAL"].copy()
            if len(manuales) > 0:
                req = manuales["REQUIERE_REVISION"].astype(str).str.lower().isin(["true", "1", "yes"])
                manuales_req = manuales[req]
                if len(manuales_req) > 0:
                    problemas.append(
                        f"⚠️ {len(manuales_req):,} programas MANUAL con REQUIERE_REVISION=True "
                        "(inconsistencia: marca False si ya revisaste)"
                    )
                else:
                    ok_msgs.append("✓ Sin programas MANUAL con revisión pendiente")

        total_ok = len(ok_msgs)
        total_warn = len(problemas)
        resumen = f"Integridad: {total_ok} OK / {total_warn} advertencia(s)\n\n"
        if ok_msgs:
            resumen += "\n".join(ok_msgs[:10]) + "\n\n"
        if problemas:
            resumen += "\n".join(problemas)

        self._log(resumen if resumen else "Sin información de integridad.")

        if total_warn == 0:
            messagebox.showinfo("✅ Integridad OK", resumen, parent=self.root)
        else:
            messagebox.showwarning("⚠️ Advertencias de integridad", resumen, parent=self.root)

    def _save(self):
        if not self.pending_updates:
            return
        if self.active_sheet == "eafit":
            messagebox.showinfo(
                "Solo lectura",
                "La hoja 'EAFIT vs Mercado' es de solo lectura.\n\n"
                "Para guardar cambios manuales, ve a 'Datos completos' o 'Programas detalle'.",
                parent=self.root,
            )
            return
        if self.active_sheet in ("total", "total_tabla"):
            keys_applied = (
                set(self.df_total["CATEGORIA_FINAL"].astype(str).str.strip()) & set(self.pending_updates.keys())
                if self.df_total is not None
                else set()
            )
        else:
            keys_applied = (
                set(self.df_detalle["CÓDIGO_SNIES_DEL_PROGRAMA"].astype(str).str.strip()) & set(self.pending_updates.keys())
                if self.df_detalle is not None
                else set()
            )
        n = len(keys_applied)
        if n == 0:
            return
        if not messagebox.askyesno(
            "Guardar cambios",
            f"¿Guardar {n} cambios en la hoja '{self.active_sheet}'?\n\nLa otra hoja no se modifica.",
            parent=self.root,
        ):
            return

        if self.active_sheet in ("total", "total_tabla"):
            # CASO A — Guardado solo de la hoja total (sin recálculo)
            try:
                with pd.ExcelWriter(self.file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    if self.df_total is not None:
                        for key in keys_applied:
                            changes = self.pending_updates[key]
                            mask = self.df_total["CATEGORIA_FINAL"].astype(str).str.strip() == str(key).strip()
                            for col, val in changes.items():
                                if col in self.df_total.columns:
                                    if col == "REQUIERE_REVISION":
                                        # Convertir "True"/"False" a bool
                                        if isinstance(val, str):
                                            val_converted = val.strip().lower() in ("true", "1", "yes", "sí", "si")
                                        else:
                                            # Evitar el bug de Python: bool("False") == True.
                                            # Normalizamos igualmente a texto y buscamos el contenido.
                                            val_str = str(val).strip().lower()
                                            val_converted = val_str in ("true", "1", "yes", "sí", "si")
                                        self.df_total.loc[mask, col] = val_converted
                                    else:
                                        self.df_total.loc[mask, col] = val
                        self.df_total.to_excel(writer, sheet_name="total", index=False)
            except Exception as e:
                if "Permission denied" in str(e) or "being used" in str(e).lower():
                    safe_messagebox_error("Error", explain_file_in_use(), parent=self.root)
                else:
                    safe_messagebox_error("Error", str(e), parent=self.root)
                return
            for k in keys_applied:
                del self.pending_updates[k]
            self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
            if not self.pending_updates:
                self.btn_save.config(state=tk.DISABLED)
            self._log(f"✓ {n} cambios guardados en {self.file_path.name}")
        else:
            # CASO B — Guardado en programas_detalle con recálculo de la hoja total en segundo plano
            if self.df_detalle is None:
                return
            # 1. Aplicar pending_updates sobre df_detalle en memoria
            for key in keys_applied:
                changes = self.pending_updates.get(key, {})
                if not changes:
                    continue
                mask = self.df_detalle["CÓDIGO_SNIES_DEL_PROGRAMA"].astype(str).str.strip() == str(key).strip()
                for col, val in changes.items():
                    if col in self.df_detalle.columns:
                        if col == "REQUIERE_REVISION":
                            # Convertir "True"/"False" a bool
                            if isinstance(val, str):
                                val_converted = val.strip().lower() in ("true", "1", "yes", "sí", "si")
                            else:
                                    # Evitar el bug de Python: bool("False") == True.
                                    # Normalizamos igualmente a texto y buscamos el contenido.
                                    val_str = str(val).strip().lower()
                                    val_converted = val_str in ("true", "1", "yes", "sí", "si")
                            self.df_detalle.loc[mask, col] = val_converted
                        else:
                            self.df_detalle.loc[mask, col] = val

            # 1b. Human-in-the-Loop: guardar correcciones de categoría en ref/feedback_manual.csv (aditivo)
            try:
                from etl.config import REF_DIR
                feedback_path = REF_DIR / "feedback_manual.csv"
                filas_feedback = []
                for key in keys_applied:
                    changes = self.pending_updates.get(key, {})
                    if "CATEGORIA_FINAL" not in changes:
                        continue
                    snies = str(key).strip()
                    cat_final = str(changes["CATEGORIA_FINAL"]).strip()
                    if snies and cat_final:
                        filas_feedback.append({"SNIES": snies, "CATEGORIA_FINAL": cat_final})
                if filas_feedback:
                    REF_DIR.mkdir(parents=True, exist_ok=True)
                    df_fb = pd.DataFrame(filas_feedback)
                    write_header = not feedback_path.exists()
                    df_fb.to_csv(
                        feedback_path,
                        mode="a",
                        index=False,
                        header=write_header,
                        encoding="utf-8-sig",
                    )
                    self._log(f"✓ {len(filas_feedback)} corrección(es) añadidas a feedback_manual.csv para retroalimentación del modelo.")
            except Exception as e:
                self._log(f"⚠️ No se pudo escribir feedback_manual.csv: {e}")

            # 2. Capturar overrides manuales existentes en la hoja total
            manual_overrides: dict[str, dict] = {}
            if self.df_total is not None and "CATEGORIA_FINAL" in self.df_total.columns and "FUENTE_CATEGORIA" in self.df_total.columns:
                mask_manual = (
                    self.df_total["FUENTE_CATEGORIA"].astype(str).str.strip().str.upper() == "MANUAL"
                )
                df_manual = self.df_total[mask_manual]
                for _, row in df_manual.iterrows():
                    cat = str(row["CATEGORIA_FINAL"]).strip()
                    if not cat:
                        continue
                    overrides: dict[str, object] = {}
                    for col in self.editable_columns:
                        if col in self.df_total.columns:
                            overrides[col] = row[col]
                    if overrides:
                        manual_overrides[cat] = overrides

            # 3. Lanzar recálculo en hilo separado
            self.btn_save.config(state=tk.DISABLED)
            for k in keys_applied:
                # limpiar solo las claves aplicadas de esta hoja
                if k in self.pending_updates:
                    del self.pending_updates[k]
            self.pending_label.config(text=f"Cambios pendientes: {len(self.pending_updates)}")
            self._log("⏳ Recalculando indicadores de la hoja total...")
            self._recalculate_total_from_detalle(keys_applied, manual_overrides)

    def _recalculate_total_from_detalle(
        self,
        keys_applied: set[str],
        manual_overrides: dict[str, dict],
    ) -> None:
        """Recalcula la hoja 'total' a partir de df_detalle usando run_fase4_desde_sabana en un hilo."""

        def _worker():
            try:
                from etl.mercado_pipeline import run_fase4_desde_sabana
                from etl.pipeline_logger import log_info

                df_sabana = self.df_detalle.copy()
                log_info("Recalculando agregación y scoring de la hoja total desde EstudioMercadoResultsPage.")
                df_total_new = run_fase4_desde_sabana(df_sabana)

                # 5. Reaplicar overrides manuales sobre el df_total recalculado
                if "CATEGORIA_FINAL" in df_total_new.columns:
                    for cat, changes in manual_overrides.items():
                        mask = df_total_new["CATEGORIA_FINAL"].astype(str).str.strip() == str(cat).strip()
                        if not mask.any():
                            continue
                        for col, val in changes.items():
                            if col in df_total_new.columns:
                                df_total_new.loc[mask, col] = val
                        if "FUENTE_CATEGORIA" in df_total_new.columns:
                            df_total_new.loc[mask, "FUENTE_CATEGORIA"] = "MANUAL"

                # 6. Escribir Excel completo con ambas hojas actualizadas
                try:
                    with pd.ExcelWriter(self.file_path, engine="openpyxl", mode="w") as writer:
                        df_total_new.to_excel(writer, sheet_name="total", index=False)
                        if self.df_detalle is not None:
                            self.df_detalle.to_excel(writer, sheet_name="programas_detalle", index=False)
                except Exception as e:
                    def _on_err():
                        if "Permission denied" in str(e) or "being used" in str(e).lower():
                            safe_messagebox_error("Error", explain_file_in_use(), parent=self.root)
                        else:
                            safe_messagebox_error("Error", str(e), parent=self.root)
                        # Rehabilitar botón de guardado si aún hay cambios
                        self.btn_save.config(state=tk.NORMAL if self.pending_updates else tk.DISABLED)

                    self.root.after(0, _on_err)
                    return

                # 7. Actualizar UI en el hilo principal
                def _on_ok():
                    self.df_total = df_total_new
                    if self.active_sheet in ("total", "total_tabla"):
                        self._apply_filter()
                    messagebox.showinfo(
                        "Guardado",
                        "Cambios guardados y hoja Total recalculada.\n"
                        f"Correcciones manuales previas conservadas: {len(manual_overrides)} categorías.",
                        parent=self.root,
                    )
                    self._log(
                        f"✓ {len(keys_applied)} cambios guardados en {self.file_path.name} "
                        f"(detalle) y hoja total recalculada"
                    )
                    self.btn_save.config(state=tk.DISABLED if not self.pending_updates else tk.NORMAL)

                self.root.after(0, _on_ok)
            except Exception as e:
                def _on_unexpected():
                    safe_messagebox_error(
                        "Error",
                        f"Error al recalcular la hoja total: {e}",
                        parent=self.root,
                    )
                    self.btn_save.config(state=tk.NORMAL if self.pending_updates else tk.DISABLED)

                self.root.after(0, _on_unexpected)

        threading.Thread(target=_worker, daemon=True).start()

    def _discard_all(self):
        if not self.pending_updates:
            return
        if not messagebox.askyesno("Descartar", "¿Descartar todos los cambios pendientes?", parent=self.root):
            return
        self.pending_updates.clear()
        self._load()

    def _open_excel(self):
        try:
            _open_in_excel(self.file_path)
        except Exception as exc:
            safe_messagebox_error("Error", str(exc), parent=self.root)

    def _prev_page(self):
        if self._filtered_df is None:
            return
        if self.page_index > 0:
            self.page_index -= 1
            self._render_page()

    def _next_page(self):
        if self._filtered_df is None:
            return
        total = len(self._filtered_df)
        max_pages = max(1, (total + self.page_size - 1) // self.page_size)
        if self.page_index < max_pages - 1:
            self.page_index += 1
            self._render_page()


def run_pipeline(
    base_dir: Path,
    log_callback=None,
    progress_callback: Callable[[int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """
    Ejecuta el pipeline completo de análisis SNIES.

    progress_callback(stage_idx, stage_name, status)
      - status: "start" | "done"
    """
    # CRÍTICO: Actualizar rutas para usar el base_dir proporcionado
    # Esto asegura que todas las rutas (outputs, ref, models, etc.) apunten al directorio correcto
    try:
        update_paths_for_base_dir(base_dir)
    except Exception as e:
        if log_callback:
            log_callback(f"[ERROR] No se pudo configurar el directorio base: {e}")
        else:
            print(f"[ERROR] No se pudo configurar el directorio base: {e}")
        return 1
    
    # Imports lazy de módulos ETL pesados (solo cuando se ejecuta el pipeline)
    import pandas as pd
    from etl.config import HISTORIC_DIR
    from etl.historicoProgramasNuevos import actualizar_historico_programas_nuevos
    from etl.normalizacion import ARCHIVO_PROGRAMAS, normalizar_programas
    from etl.normalizacion_final import aplicar_normalizacion_final
    from etl.pipeline_logger import (
        log_error,
        log_etapa_completada,
        log_etapa_iniciada,
        log_exception,
        log_fin,
        log_info,
        log_inicio,
        log_warning,
    )
    from etl.procesamientoSNIES import procesar_programas_nuevos
    from etl.clasificacionProgramas import clasificar_programas_nuevos
    
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(stage_idx: int, stage_name: str, status: str):
        if progress_callback is None:
            return
        try:
            progress_callback(stage_idx, stage_name, status)
        except Exception:
            pass

    # Lock file: indica que el pipeline está ejecutándose.
    # Sirve para que otras ventanas no intenten escribir Programas.xlsx mientras se reescribe.
    lock_file = ARCHIVO_PROGRAMAS.parent / ".pipeline.lock"
    
    tiempo_inicio = time.time()
    log_inicio()
    # Crear lock
    lock_created = False
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text(f"running_since={time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        lock_created = True
    except Exception as e:
        # Si no se puede crear el lock, no bloqueamos la ejecución pero lo registramos
        log_warning(f"No se pudo crear lock file: {e}")

    pipeline_failed = [False]

    try:
        # Pre-checks centralizados (fallar temprano con mensajes claros)
        ok_env, mensajes_env = validar_entorno_pipeline()
        if not ok_env:
            msg = "El entorno no está listo para ejecutar el pipeline:\n\n" + "\n".join(f"• {m}" for m in mensajes_env if "Todo listo" not in m)
            log(f"[ERROR] {msg}")
            log_error(msg)
            pipeline_failed[0] = True
            return 1
        log(mensajes_env[0] if mensajes_env else "Pre-checks OK.")

        # Verificar modelos ML y entrenar automáticamente si es la primera vez
        modelo_clf = None
        necesita_entrenar = False
        try:
            modelo_clf = MODELS_DIR / "clasificador_referentes.pkl"
            modelo_emb = MODELS_DIR / "modelo_embeddings.pkl"
            encoder = MODELS_DIR / "encoder_programas_eafit.pkl"
            
            # Verificar si faltan modelos (primera ejecución)
            if not modelo_clf.exists() or not modelo_emb.exists() or not encoder.exists():
                necesita_entrenar = True
                log("[INFO] No se encontraron modelos ML entrenados. Esto parece ser la primera ejecución.")
                log("[INFO] Se intentará entrenar el modelo automáticamente...")
        except Exception:
            pass

        progress(0, "Inicializando", "start")
        log("=== Paso 1: Resguardo de históricos ===")
        # Importar HISTORIC_DIR aquí para asegurar que esté disponible después de set_base_dir
        from etl.config import HISTORIC_DIR
        log(
            f"Si se logra obtener una versión nueva de Programas.xlsx, "
            f"el archivo anterior se trasladará a: {HISTORIC_DIR}"
        )
        log(
            "Si falla la descarga SNIES, no se realizará ninguna modificación sobre archivos existentes."
        )
        progress(0, "Inicializando", "done")
    
        # Verificar cancelación antes de iniciar descarga
        if cancel_event and cancel_event.is_set():
            log("[CANCELADO] Pipeline cancelado antes de iniciar descarga.")
            return 1
        
        progress(1, "Descarga SNIES", "start")
        t_etapa = time.time()
        log("=== Paso 2: Descarga de Programas SNIES ===")
        log_etapa_iniciada("Descarga SNIES")
        log("Descargando archivo desde SNIES...")
        # Import diferido: permite abrir el menú aunque no estén instaladas las dependencias de Selenium.
        try:
            from etl.descargaSNIES import main as descargar_programas
        except Exception as exc:
            error_msg = (
                "No se pudo iniciar el módulo de descarga (Selenium). "
                "Verifica que el entorno virtual esté activo y que las dependencias estén instaladas.\n\n"
                f"Detalle: {exc}"
            )
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            pipeline_failed[0] = True
            return 1

        ruta_descargada = descargar_programas(log_callback=log, cancel_event=cancel_event)
        
        # Verificar cancelación después de descarga
        if cancel_event and cancel_event.is_set():
            log("[CANCELADO] Pipeline cancelado durante la descarga.")
            pipeline_failed[0] = True
            return 1
        
        if not ruta_descargada:
            error_msg = "No se obtuvo una ruta de descarga válida."
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            pipeline_failed[0] = True
            return 1

        ruta_descargada = Path(ruta_descargada)
        if not ruta_descargada.exists():
            error_msg = f"El archivo descargado no existe: {ruta_descargada}"
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            return 1

        nombre_archivo = ruta_descargada.name
        log(f"✓ Archivo descargado: {nombre_archivo}")
        
        # Verificar si había un Programas.xlsx anterior que fue movido a histórico
        from etl.config import HISTORIC_DIR
        archivos_historicos_recientes = sorted(
            HISTORIC_DIR.glob("Programas_*.xlsx"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:1]  # Solo el más reciente
        if archivos_historicos_recientes:
            archivo_historico_reciente = archivos_historicos_recientes[0]
            log(f"✓ Archivo anterior movido a histórico: {archivo_historico_reciente.name}")
        
        log_etapa_completada("Descarga SNIES", f"{nombre_archivo} (duración: {time.time() - t_etapa:.1f}s)")
        progress(1, "Descarga SNIES", "done")

        # Validación de schema mínimo (para detectar cambios en SNIES)
        ok, msg = validate_programas_schema(ARCHIVO_PROGRAMAS)
        if not ok:
            log(f"[ERROR] {msg}")
            log_error(msg)
            pipeline_failed[0] = True
            return 1

        if ruta_descargada != ARCHIVO_PROGRAMAS:
            warning_msg = (
                f"El archivo descargado está en {ruta_descargada}, "
                f"pero la normalización usará {ARCHIVO_PROGRAMAS}."
            )
            log(f"[WARN] {warning_msg}")
            log_warning(warning_msg)

        # OPTIMIZACIÓN: Leer archivo una sola vez y trabajar en memoria
        log("=== OPTIMIZACIÓN: Pipeline en memoria ===")
        log("Leyendo archivo una vez y procesando en memoria para reducir I/O...")
        from etl.exceptions_helpers import leer_excel_con_reintentos
        
        try:
            df_programas = leer_excel_con_reintentos(ARCHIVO_PROGRAMAS, sheet_name="Programas")
            
            # Validar que el DataFrame no esté vacío
            if df_programas is None:
                error_msg = "El archivo Programas.xlsx está vacío o no se pudo leer correctamente."
                log(f"[ERROR] {error_msg}")
                log_error(error_msg)
                pipeline_failed[0] = True
                return 1
            
            if len(df_programas) == 0:
                error_msg = "El archivo Programas.xlsx no contiene filas de datos."
                log(f"[ERROR] {error_msg}")
                log_error(error_msg)
                pipeline_failed[0] = True
                return 1
            
            log(f"✓ Archivo cargado en memoria: {len(df_programas)} filas")
        except Exception as exc:
            error_msg = f"No se pudo leer Programas.xlsx: {exc}"
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            log_exception(exc)
            pipeline_failed[0] = True
            return 1

        # Verificar cancelación antes de normalización
        if cancel_event and cancel_event.is_set():
            log("[CANCELADO] Pipeline cancelado antes de normalización.")
            pipeline_failed[0] = True
            return 1
        
        progress(2, "Normalización", "start")
        t_etapa = time.time()
        log("=== Paso 3: Normalización de columnas ===")
        log_etapa_iniciada("Normalización")
        try:
            log("Normalizando columnas del archivo...")
            df_programas = normalizar_programas(df=df_programas)  # Modo optimizado: en memoria
            
            # Validar que la normalización retornó un DataFrame válido
            if df_programas is None or len(df_programas) == 0:
                error_msg = "La normalización retornó un DataFrame vacío o None."
                log(f"[ERROR] {error_msg}")
                log_error(error_msg)
                pipeline_failed[0] = True
                return 1
            
            log("✓ Normalización completada")
            log_etapa_completada("Normalización", f"duración: {time.time() - t_etapa:.1f}s")
            progress(2, "Normalización", "done")
        except Exception as exc:
            error_msg = f"Falló la normalización: {exc}"
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            log_exception(exc)
            pipeline_failed[0] = True
            return 1

        # Verificar cancelación antes de procesamiento
        if cancel_event and cancel_event.is_set():
            log("[CANCELADO] Pipeline cancelado antes de procesamiento de programas nuevos.")
            pipeline_failed[0] = True
            return 1
        
        progress(3, "Programas nuevos", "start")
        t_etapa = time.time()
        log("=== Paso 4: Procesamiento de programas nuevos ===")
        log_etapa_iniciada("Procesamiento de programas nuevos")
        try:
            log("Procesando programas nuevos...")
            df_programas = procesar_programas_nuevos(df=df_programas)  # Modo optimizado: en memoria
            
            # Validar que el procesamiento retornó un DataFrame válido
            if df_programas is None or len(df_programas) == 0:
                error_msg = "El procesamiento de programas nuevos retornó un DataFrame vacío o None."
                log(f"[ERROR] {error_msg}")
                log_error(error_msg)
                pipeline_failed[0] = True
                return 1
            
            # Verificar si hay programas nuevos detectados
            if "PROGRAMA_NUEVO" in df_programas.columns:
                # Contar programas nuevos usando el mismo patrón que se usa en otras partes del código
                programas_nuevos = df_programas[
                    df_programas["PROGRAMA_NUEVO"].astype(str).str.strip().str.upper() == "SÍ"
                ]
                cantidad_nuevos = len(programas_nuevos)
                
                log(f"Programas nuevos detectados: {cantidad_nuevos}")
                
                if cantidad_nuevos == 0:
                    # No hay programas nuevos, cargar la última ejecución correcta del histórico
                    info_msg = (
                        "No se han detectado programas nuevos después de comparar con los archivos históricos.\n\n"
                        "Esto significa que todos los programas en el archivo descargado ya estaban presentes "
                        "en ejecuciones anteriores del pipeline.\n\n"
                        "Se cargará la última ejecución correcta del histórico para continuar trabajando con esos datos."
                    )
                    log(f"[INFO] {info_msg}")
                    log_info(info_msg)
                    
                    # Intentar cargar el último archivo histórico
                    from etl.procesamientoSNIES import obtener_ultimo_archivo_historico
                    from etl.exceptions_helpers import leer_excel_con_reintentos
                    
                    archivo_historico = obtener_ultimo_archivo_historico(HISTORIC_DIR)
                    if archivo_historico and archivo_historico.exists():
                        try:
                            log(f"Cargando última ejecución correcta desde: {archivo_historico.name}")
                            df_programas = leer_excel_con_reintentos(archivo_historico, sheet_name="Programas")
                            log(f"✓ Archivo histórico cargado: {len(df_programas)} programas")
                            log_info(f"Cargado archivo histórico: {archivo_historico.name} ({len(df_programas)} programas)")
                        except Exception as e:
                            error_msg = f"No se pudo cargar el archivo histórico: {e}"
                            log(f"[ERROR] {error_msg}")
                            log_error(error_msg)
                            log("Continuando con el archivo actual (sin programas nuevos)...")
                    else:
                        log("⚠️ No se encontró archivo histórico. Continuando con el archivo actual (sin programas nuevos)...")
                        log_warning("No se encontró archivo histórico para cargar")
                    
                    log("El pipeline continuará con los datos disponibles (sin programas nuevos para clasificar).")
                    log_etapa_completada("Procesamiento de programas nuevos", f"duración: {time.time() - t_etapa:.1f}s")
                    progress(3, "Programas nuevos", "done")
                    # Continuar con el pipeline en lugar de detenerse
                else:
                    log(f"✓ Procesamiento completado: {cantidad_nuevos} programa(s) nuevo(s) detectado(s)")
                    log_etapa_completada("Procesamiento de programas nuevos", f"duración: {time.time() - t_etapa:.1f}s")
                    progress(3, "Programas nuevos", "done")
            else:
                log("⚠️ Advertencia: No se encontró la columna PROGRAMA_NUEVO. Continuando con precaución...")
                log_etapa_completada("Procesamiento de programas nuevos", f"duración: {time.time() - t_etapa:.1f}s")
                progress(3, "Programas nuevos", "done")
        except Exception as exc:
            error_msg = f"Falló el procesamiento de programas nuevos: {exc}"
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            pipeline_failed[0] = True
            return 1

        progress(4, "Clasificación", "start")
        t_etapa_clasif = time.time()
        log("=== Paso 5: Clasificación de programas nuevos ===")
        log_etapa_iniciada("Clasificación de programas nuevos")
        
        # Si es la primera ejecución y faltan modelos, entrenar automáticamente
        if necesita_entrenar:
            log("Entrenando modelo ML automáticamente (primera ejecución)...")
            log("Esto puede tardar varios minutos. Por favor espera...")
            try:
                from etl.clasificacionProgramas import entrenar_y_guardar_modelo, get_archivo_referentes
                
                # Verificar que existe el archivo de referentes
                archivo_referentes = get_archivo_referentes()
                if not archivo_referentes or not archivo_referentes.exists():
                    error_msg = (
                        f"No se encontró el archivo de referentes para entrenar.\n\n"
                        "Coloca el archivo referentesUnificados.csv o .xlsx en la carpeta ref/ "
                        "y vuelve a ejecutar el pipeline."
                    )
                    log(f"[ERROR] {error_msg}")
                    log_error(error_msg)
                    log("Continuando sin clasificación...")
                    progress(4, "Clasificación", "done")
                else:
                    entrenar_y_guardar_modelo()
                    log("✓ Modelo entrenado exitosamente")
            except Exception as exc:
                error_msg = f"Falló el entrenamiento automático del modelo: {exc}"
                log(f"[WARN] {error_msg}")
                log("Continuando sin clasificación...")
                log_error(error_msg)
                log_warning("Continuando sin clasificación...")
                progress(4, "Clasificación", "done")
        
        # Intentar clasificar programas nuevos
        try:
            log("Clasificando programas nuevos...")
            log("La clasificación compara cada programa nuevo del SNIES con el catálogo EAFIT.")
            log("Para cada programa nuevo, determina:")
            log("  - Si es referente (ES_REFERENTE: Sí/No)")
            log("  - Qué programa EAFIT le corresponde (PROGRAMA_EAFIT_CODIGO y PROGRAMA_EAFIT_NOMBRE)")
            log("  - La probabilidad de que sea referente (PROBABILIDAD: 0.0-1.0)")
            log("  - Métricas de similitud (SIMILITUD_EMBEDDING, SIMILITUD_CAMPO, SIMILITUD_NIVEL)")
            def _progress_clasif(cur: int, tot: int, nom: str) -> None:
                log(f"  Clasificando {cur}/{tot}: {nom[:50]}{'...' if len(nom) > 50 else ''}")
            df_programas = clasificar_programas_nuevos(df_programas=df_programas, progress_callback=_progress_clasif)  # Modo optimizado: en memoria
            
            # Validar que la clasificación retornó un DataFrame válido
            if df_programas is None:
                error_msg = "La clasificación retornó None."
                log(f"[WARN] {error_msg}")
                log_warning(error_msg)
                # Intentar cargar el último histórico
                log("[INFO] Intentando cargar último histórico...")
                from etl.procesamientoSNIES import obtener_ultimo_archivo_historico
                from etl.exceptions_helpers import leer_excel_con_reintentos
                
                archivo_historico = obtener_ultimo_archivo_historico(HISTORIC_DIR)
                if archivo_historico and archivo_historico.exists():
                    try:
                        log(f"Cargando última ejecución correcta desde: {archivo_historico.name}")
                        df_programas = leer_excel_con_reintentos(archivo_historico, sheet_name="Programas")
                        log(f"✓ Archivo histórico cargado: {len(df_programas)} programas")
                    except Exception as e:
                        error_msg = f"No se pudo cargar el archivo histórico: {e}"
                        log(f"[WARN] {error_msg}")
                        log_warning(error_msg)
                        progress(4, "Clasificación", "done")
                        # Continuar sin clasificación
                else:
                    log("[WARN] No se encontró archivo histórico. Continuando sin clasificación...")
                    progress(4, "Clasificación", "done")
            elif len(df_programas) == 0:
                # Si no hay programas nuevos, clasificar_programas_nuevos retorna el DataFrame completo
                # Si aún así está vacío, intentar cargar el último histórico
                log("[INFO] No hay programas para procesar. Intentando cargar último histórico...")
                from etl.procesamientoSNIES import obtener_ultimo_archivo_historico
                from etl.exceptions_helpers import leer_excel_con_reintentos
                
                archivo_historico = obtener_ultimo_archivo_historico(HISTORIC_DIR)
                if archivo_historico and archivo_historico.exists():
                    try:
                        log(f"Cargando última ejecución correcta desde: {archivo_historico.name}")
                        df_programas = leer_excel_con_reintentos(archivo_historico, sheet_name="Programas")
                        log(f"✓ Archivo histórico cargado: {len(df_programas)} programas")
                    except Exception as e:
                        error_msg = f"No se pudo cargar el archivo histórico: {e}"
                        log(f"[WARN] {error_msg}")
                        log_warning(error_msg)
                        progress(4, "Clasificación", "done")
                        # Continuar sin clasificación
                else:
                    log("[WARN] No se encontró archivo histórico. Continuando sin clasificación...")
                    progress(4, "Clasificación", "done")
            else:
                log("✓ Clasificación completada")
                log_etapa_completada("Clasificación de programas nuevos", f"duración: {time.time() - t_etapa_clasif:.1f}s")
                progress(4, "Clasificación", "done")
        except FileNotFoundError as exc:
            # Error específico cuando faltan modelos o catálogo EAFIT
            error_msg = f"Error crítico en clasificación: {exc}"
            log(f"[ERROR] {error_msg}")
            log("=" * 60)
            log("IMPORTANTE: Sin el catálogo EAFIT o los modelos ML, NO se pueden identificar referentes.")
            log("Verifica que:")
            log("  1. El archivo 'catalogoOfertasEAFIT.xlsx' o '.csv' esté en ref/backup/ o ref/")
            log("  2. Los modelos ML estén entrenados y disponibles en models/")
            log("=" * 60)
            log_error(error_msg)
            log_warning("Continuando sin clasificación...")
            progress(4, "Clasificación", "done")
        except ValueError as exc:
            # Error de validación (catálogo vacío, columnas faltantes, etc.)
            error_msg = f"Error de validación en clasificación: {exc}"
            log(f"[ERROR] {error_msg}")
            log("=" * 60)
            log("IMPORTANTE: El catálogo EAFIT no es válido o está vacío.")
            log("Sin un catálogo válido, NO se pueden identificar referentes.")
            log("=" * 60)
            log_error(error_msg)
            log_warning("Continuando sin clasificación...")
            progress(4, "Clasificación", "done")
        except Exception as exc:
            error_msg = f"Falló la clasificación de programas nuevos: {exc}"
            log(f"[ERROR] {error_msg}")
            log("=" * 60)
            log("Detalles del error:")
            import traceback
            log(traceback.format_exc())
            log("=" * 60)
            log_error(error_msg)
            log_warning("Continuando sin clasificación...")
            progress(4, "Clasificación", "done")
            # No retornamos error aquí porque la clasificación es opcional, pero registramos el error completo

        # Verificar cancelación antes de normalización final
        if cancel_event and cancel_event.is_set():
            log("[CANCELADO] Pipeline cancelado antes de normalización final.")
            pipeline_failed[0] = True
            return 1
        
        progress(5, "Normalización final", "start")
        t_etapa = time.time()
        log("=== Paso 6: Normalización final de ortografía y formato ===")
        log_etapa_iniciada("Normalización final")
        try:
            log("Aplicando normalización final...")
            df_programas = aplicar_normalizacion_final(df=df_programas)  # Modo optimizado: en memoria
            
            # Validar que la normalización final retornó un DataFrame válido
            if df_programas is None or len(df_programas) == 0:
                error_msg = "La normalización final retornó un DataFrame vacío o None."
                log(f"[ERROR] {error_msg}")
                log_error(error_msg)
                pipeline_failed[0] = True
                return 1
            
            log("✓ Normalización final completada")
            log_etapa_completada("Normalización final", f"duración: {time.time() - t_etapa:.1f}s")
            progress(5, "Normalización final", "done")
        except Exception as exc:
            error_msg = f"Falló la normalización final: {exc}"
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            log_exception(exc)
            pipeline_failed[0] = True
            return 1

        # Verificar cancelación antes de guardar
        if cancel_event and cancel_event.is_set():
            log("[CANCELADO] Pipeline cancelado antes de guardar archivo.")
            pipeline_failed[0] = True
            return 1
        
        # OPTIMIZACIÓN: Escribir archivo una sola vez al final
        progress(6, "Guardando archivo", "start")
        t_etapa = time.time()
        log("=== Paso 7: Guardando archivo final ===")
        log_etapa_iniciada("Guardado de archivo")
        try:
            from etl.exceptions_helpers import escribir_excel_con_reintentos
            log("Guardando Programas.xlsx con todos los cambios aplicados...")
            escribir_excel_con_reintentos(ARCHIVO_PROGRAMAS, df_programas, sheet_name="Programas")
            log("✓ Archivo guardado exitosamente")
            log_etapa_completada("Guardado de archivo", f"duración: {time.time() - t_etapa:.1f}s")
            progress(6, "Guardando archivo", "done")
        except Exception as exc:
            error_msg = f"Falló el guardado del archivo: {exc}"
            log(f"[ERROR] {error_msg}")
            log_error(error_msg)
            log_exception(exc)
            pipeline_failed[0] = True
            return 1

        progress(7, "Histórico programas nuevos", "start")
        t_etapa = time.time()
        log("=== Paso 8: Actualización de histórico de programas nuevos ===")
        log_etapa_iniciada("Actualización de histórico de programas nuevos")
        try:
            log("Actualizando histórico...")
            actualizar_historico_programas_nuevos()
            log("✓ Histórico actualizado")
            log_etapa_completada("Actualización de histórico de programas nuevos", f"duración: {time.time() - t_etapa:.1f}s")
            progress(7, "Histórico programas nuevos", "done")
        except Exception as exc:
            error_msg = f"Falló la actualización del histórico: {exc}"
            log(f"[WARN] {error_msg}")
            log_error(error_msg)
            progress(7, "Histórico programas nuevos", "done")
            # No retornamos error aquí porque el histórico es complementario

        tiempo_fin = time.time()
        duracion_minutos = (tiempo_fin - tiempo_inicio) / 60.0
        log("Pipeline completado exitosamente.")
        log_fin(duracion_minutos)
        try:
            from etl.config import set_last_success
            from datetime import datetime
            set_last_success(datetime.now().isoformat(), duracion_minutos)
        except Exception:
            pass

        return 0
    finally:
        # Remover lock siempre (incluso si hubo KeyboardInterrupt o SystemExit)
        try:
            if lock_file.exists():
                lock_file.unlink()
                if lock_created:
                    log_info("Lock file eliminado correctamente.")
        except Exception as e:
            log_error(f"No se pudo eliminar lock file: {e}")
            # Intentar una vez más después de un breve delay
            try:
                import time as time_module
                time_module.sleep(0.5)
                if lock_file.exists():
                    lock_file.unlink()
            except Exception:
                pass


def main():
    """Función principal que inicia el Menú Principal."""
    root = tk.Tk()
    app = MainMenuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
